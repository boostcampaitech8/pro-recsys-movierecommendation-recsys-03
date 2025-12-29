"""
Data loading and preprocessing module for EASE model.
Handles data conversion, train/valid split, and ID mapping.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple, Dict, List, Optional
import random
from collections import defaultdict


class DataLoader:
    """
    DataLoader for movie recommendation task.
    Handles ID mapping, train/valid split, and sparse matrix creation.
    """
    
    def __init__(self, train_file: str, seed: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            train_file: Path to train_ratings.csv
            seed: Random seed for reproducibility
        """
        self.train_file = train_file
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # ID mappings (original ID -> internal index)
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}
        
        # Data
        self.n_users: int = 0
        self.n_items: int = 0
        self.user_sequences: Dict[int, List[Tuple[int, int]]] = {}  # user_idx -> [(item_idx, timestamp), ...]
        
        # Load and process data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the rating data."""
        print("Loading data...")
        df = pd.read_csv(self.train_file)
        
        # Create ID mappings
        unique_users = df['user'].unique()
        unique_items = df['item'].unique()
        
        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        print(f"  Users: {self.n_users}, Items: {self.n_items}")
        print(f"  Interactions: {len(df)}")
        
        # Build user sequences (sorted by timestamp)
        self.user_sequences = {}
        for user_id, group in df.groupby('user'):
            user_idx = self.user2idx[user_id]
            # Sort by timestamp and store (item_idx, timestamp)
            sorted_items = group.sort_values('time')[['item', 'time']].values
            self.user_sequences[user_idx] = [
                (self.item2idx[int(item)], int(time)) 
                for item, time in sorted_items
            ]
    
    def create_train_valid_split(
        self, 
        valid_random_items: int = 9, 
        valid_seq_items: int = 1
    ) -> Tuple[csr_matrix, csr_matrix, Dict[int, List[int]]]:
        """
        Create train/valid split mimicking the competition's data generation.
        
        Competition removes:
        - Some sequential items (next items)
        - Some random items from the sequence
        
        Args:
            valid_random_items: Number of random items to hold out per user
            valid_seq_items: Number of sequential (last) items to hold out per user
            
        Returns:
            train_matrix: User-item interaction matrix for training
            valid_matrix: User-item interaction matrix (train items, for filtering)
            valid_ground_truth: Dict of user_idx -> list of held-out item indices
        """
        print(f"Creating train/valid split...")
        print(f"  Hold out: {valid_seq_items} sequential + {valid_random_items} random items per user")
        
        train_interactions = []
        valid_ground_truth = {}
        
        for user_idx, seq in self.user_sequences.items():
            items_with_time = seq.copy()
            n_items = len(items_with_time)
            
            # Need enough items for valid split
            total_holdout = valid_random_items + valid_seq_items
            if n_items <= total_holdout:
                # Keep all for training if sequence too short
                for item_idx, _ in items_with_time:
                    train_interactions.append((user_idx, item_idx))
                valid_ground_truth[user_idx] = []
                continue
            
            # Hold out last N items (sequential)
            seq_holdout_items = [item_idx for item_idx, _ in items_with_time[-valid_seq_items:]]
            remaining = items_with_time[:-valid_seq_items]
            
            # Hold out random items from remaining
            if len(remaining) > valid_random_items:
                random_indices = random.sample(range(len(remaining)), valid_random_items)
                random_holdout_items = [remaining[i][0] for i in random_indices]
                train_items = [remaining[i][0] for i in range(len(remaining)) if i not in random_indices]
            else:
                random_holdout_items = []
                train_items = [item_idx for item_idx, _ in remaining]
            
            # Store results
            for item_idx in train_items:
                train_interactions.append((user_idx, item_idx))
            
            valid_ground_truth[user_idx] = seq_holdout_items + random_holdout_items
        
        # Create sparse matrices
        train_rows = [u for u, i in train_interactions]
        train_cols = [i for u, i in train_interactions]
        train_data = [1.0] * len(train_interactions)
        
        train_matrix = csr_matrix(
            (train_data, (train_rows, train_cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        # Valid matrix is same as train (for filtering already seen items)
        valid_matrix = train_matrix.copy()
        
        n_valid_items = sum(len(v) for v in valid_ground_truth.values())
        print(f"  Train interactions: {len(train_interactions)}")
        print(f"  Valid items to predict: {n_valid_items}")
        
        return train_matrix, valid_matrix, valid_ground_truth
    
    def create_submission_matrix(self) -> csr_matrix:
        """
        Create full interaction matrix for submission (no holdout).
        
        Returns:
            Full user-item interaction matrix
        """
        print("Creating submission matrix (full data)...")
        
        rows, cols, data = [], [], []
        for user_idx, seq in self.user_sequences.items():
            for item_idx, _ in seq:
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(1.0)
        
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        print(f"  Total interactions: {len(data)}")
        return matrix
    
    def get_user_sequence_with_time(self, user_idx: int) -> List[Tuple[int, int]]:
        """Get user's item sequence with timestamps."""
        return self.user_sequences.get(user_idx, [])
    
    def get_original_user_id(self, user_idx: int) -> int:
        """Convert internal user index to original user ID."""
        return self.idx2user[user_idx]
    
    def get_original_item_id(self, item_idx: int) -> int:
        """Convert internal item index to original item ID."""
        return self.idx2item[item_idx]


class WeightedMatrixBuilder:
    """
    Build weighted co-occurrence matrices based on session/page hierarchy.
    
    Hierarchy:
        User Sequence → Session (30min gap) → Page (30sec gap) → Items
    
    Weight rules:
        - Same page: within_page_weight (default 1.0)
        - Same session, different page: exp(-Δt / cross_page_tau)
        - Different session: 0 (not computed)
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def _split_into_sessions(
        self,
        sequence: List[Tuple[int, int]],
        session_threshold: float = 1800.0
    ) -> List[List[Tuple[int, int]]]:
        """
        Split a sequence into sessions based on time gap.
        
        Args:
            sequence: User's (item_idx, timestamp) sequence (sorted by time)
            session_threshold: Session split threshold in seconds (default 30 min)
        
        Returns:
            List of sessions, each session is a list of (item_idx, timestamp)
        """
        if len(sequence) == 0:
            return []
        
        sessions = []
        current_session = [sequence[0]]
        
        for i in range(1, len(sequence)):
            prev_time = sequence[i - 1][1]
            curr_time = sequence[i][1]
            gap = curr_time - prev_time
            
            if gap > session_threshold:
                # Start new session
                sessions.append(current_session)
                current_session = [sequence[i]]
            else:
                current_session.append(sequence[i])
        
        # Add last session
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _split_into_pages(
        self,
        session: List[Tuple[int, int]],
        page_threshold: float = 30.0
    ) -> List[List[Tuple[int, int]]]:
        """
        Split a session into pages based on consecutive time gap.
        
        Args:
            session: One session's (item_idx, timestamp) list (sorted by time)
            page_threshold: Page split threshold in seconds (default 30 sec)
        
        Returns:
            List of pages, each page is a list of (item_idx, timestamp)
        """
        if len(session) == 0:
            return []
        
        pages = []
        current_page = [session[0]]
        
        for i in range(1, len(session)):
            prev_time = session[i - 1][1]
            curr_time = session[i][1]
            gap = curr_time - prev_time
            
            if gap > page_threshold:
                # Start new page
                pages.append(current_page)
                current_page = [session[i]]
            else:
                current_page.append(session[i])
        
        # Add last page
        if current_page:
            pages.append(current_page)
        
        return pages
    
    def build_session_cooccurrence_matrix(
        self,
        session_threshold: float = 1800.0,
        page_threshold: float = 30.0,
        within_page_weight: float = 1.0,
        cross_page_tau: float = 60.0,
        train_matrix: Optional[csr_matrix] = None,
        verbose: bool = True
    ) -> csr_matrix:
        """
        Build session/page-based Item-Item Co-occurrence matrix.
        
        Hierarchy:
            User Sequence → Session (session_threshold gap) → Page (page_threshold gap) → Items
        
        Weight rules:
            - Same page: within_page_weight (default 1.0)
            - Same session, different page: exp(-Δt / cross_page_tau)
            - Different session: 0 (not computed)
        
        Args:
            session_threshold: Session split threshold in seconds (default 30 min)
            page_threshold: Page split threshold in seconds (default 30 sec)
            within_page_weight: Weight for item pairs within same page
            cross_page_tau: Time decay parameter for cross-page pairs
            train_matrix: If provided, only consider items in training set
            verbose: Print progress
        
        Returns:
            Item-Item co-occurrence matrix (sparse)
        """
        # Cross-page weight cutoff: exp(-5) ≈ 0.0067, negligible
        max_time_for_weight = 5.0 * cross_page_tau
        
        if verbose:
            print(f"Building session-based co-occurrence matrix...")
            print(f"  Session threshold: {session_threshold}s ({session_threshold/60:.1f} min)")
            print(f"  Page threshold: {page_threshold}s")
            print(f"  Within-page weight: {within_page_weight}")
            print(f"  Cross-page tau: {cross_page_tau}s")
            print(f"  Cross-page max time: {max_time_for_weight}s (5*tau cutoff)")
        
        n_items = self.data_loader.n_items
        
        # Use dictionary for sparse construction
        cooc_dict = defaultdict(float)
        
        # Optimization: Pre-build allowed items set per user (O(1) lookup vs O(log n) CSR access)
        train_items_by_user = None
        if train_matrix is not None:
            train_csr = train_matrix.tocsr()
            train_items_by_user = {
                user_idx: set(train_csr[user_idx].indices)
                for user_idx in range(train_csr.shape[0])
            }
        
        # Statistics
        total_sessions = 0
        total_pages = 0
        within_page_pairs = 0
        cross_page_pairs = 0
        skipped_by_cutoff = 0
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            if len(seq) < 2:
                continue
            
            # Filter to training items if needed (optimized with set lookup)
            if train_items_by_user is not None:
                allowed_items = train_items_by_user.get(user_idx, set())
                seq = [(item_idx, t) for item_idx, t in seq if item_idx in allowed_items]
            
            if len(seq) < 2:
                continue
            
            # Split into sessions
            sessions = self._split_into_sessions(seq, session_threshold)
            total_sessions += len(sessions)
            
            for session in sessions:
                if len(session) < 2:
                    continue
                
                # Split session into pages
                pages = self._split_into_pages(session, page_threshold)
                total_pages += len(pages)
                
                # 1. Within-page pairs: all pairs get within_page_weight
                for page in pages:
                    if len(page) < 2:
                        continue
                    
                    items_in_page = [item_idx for item_idx, _ in page]
                    
                    for i in range(len(items_in_page)):
                        for j in range(i + 1, len(items_in_page)):
                            item_i = items_in_page[i]
                            item_j = items_in_page[j]
                            
                            # Symmetric key (smaller index first)
                            pair_key = (min(item_i, item_j), max(item_i, item_j))
                            cooc_dict[pair_key] += within_page_weight
                            within_page_pairs += 1
                
                # 2. Cross-page pairs: exp(-Δt / tau) with cutoff optimization
                if len(pages) >= 2:
                    # Process page pairs directly instead of flattening all items
                    for page_i_idx in range(len(pages)):
                        for page_j_idx in range(page_i_idx + 1, len(pages)):
                            page_i = pages[page_i_idx]
                            page_j = pages[page_j_idx]
                            
                            # Early termination: check min possible time diff between pages
                            # page_i's last item vs page_j's first item (pages are time-sorted)
                            min_time_diff = page_j[0][1] - page_i[-1][1]
                            if min_time_diff > max_time_for_weight:
                                # All subsequent pages will have even larger time diff
                                skipped_by_cutoff += len(page_j) * len(page_i)
                                continue
                            
                            # Calculate pairs between page_i and page_j
                            for item_i, time_i in page_i:
                                for item_j, time_j in page_j:
                                    dt = abs(time_j - time_i)
                                    
                                    # Skip if time diff exceeds cutoff
                                    if dt > max_time_for_weight:
                                        skipped_by_cutoff += 1
                                        continue
                                    
                                    weight = np.exp(-dt / cross_page_tau)
                                    
                                    # Symmetric key
                                    pair_key = (min(item_i, item_j), max(item_i, item_j))
                                    cooc_dict[pair_key] += weight
                                    cross_page_pairs += 1
        
        if verbose:
            print(f"  Total sessions: {total_sessions:,}")
            print(f"  Total pages: {total_pages:,}")
            print(f"  Within-page pairs: {within_page_pairs:,}")
            print(f"  Cross-page pairs: {cross_page_pairs:,}")
            print(f"  Skipped by cutoff: {skipped_by_cutoff:,}")
            print(f"  Unique item pairs: {len(cooc_dict):,}")
            print(f"  Converting to sparse matrix...")
        
        # Convert to sparse matrix
        rows, cols, data = [], [], []
        for (i, j), weight in cooc_dict.items():
            # Add both directions for symmetric matrix
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([weight, weight])
        
        cooc_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_items, n_items),
            dtype=np.float32
        )
        
        if verbose:
            print(f"  Matrix nnz: {cooc_matrix.nnz:,}")
        
        return cooc_matrix
    
    def build_combined_cooccurrence_matrix(
        self,
        train_matrix: csr_matrix,
        alpha: float = 0.3,
        session_threshold: float = 1800.0,
        page_threshold: float = 30.0,
        within_page_weight: float = 1.0,
        cross_page_tau: float = 60.0,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Build combined co-occurrence matrix using Additive approach.
        
        C_final = X^T X + alpha * scale * normalize(C_session)
        
        This preserves the base matrix (X^T X) completely and adds
        normalized session-based information on top.
        
        Args:
            train_matrix: User-item interaction matrix
            alpha: Weight for session-based matrix (0 = base EASE only)
            session_threshold: Session split threshold in seconds
            page_threshold: Page split threshold in seconds
            within_page_weight: Weight for same-page pairs
            cross_page_tau: Time decay parameter for cross-page pairs
            verbose: Print progress
            
        Returns:
            Combined co-occurrence matrix (dense, for EASE)
        """
        if verbose:
            print(f"Building combined co-occurrence matrix...")
            print(f"  Alpha (session weight): {alpha}")
        
        n_items = self.data_loader.n_items
        
        # Step 1: Build base co-occurrence matrix (X^T X)
        if verbose:
            print(f"\n  [1/3] Computing base matrix (X^T X)...")
        C_base = (train_matrix.T @ train_matrix).toarray().astype(np.float32)
        
        base_mean = C_base[C_base > 0].mean()
        base_max = C_base.max()
        
        if verbose:
            print(f"    Base matrix: shape={C_base.shape}, nnz={np.count_nonzero(C_base):,}")
            print(f"    Base stats: mean={base_mean:.2f}, max={base_max:.2f}")
        
        # If alpha is 0, just return base matrix
        if alpha == 0:
            if verbose:
                print(f"\n  Alpha=0, returning base matrix only.")
            return C_base
        
        # Step 2: Build session-based co-occurrence matrix
        if verbose:
            print(f"\n  [2/3] Computing session-based matrix...")
        C_session = self.build_session_cooccurrence_matrix(
            session_threshold=session_threshold,
            page_threshold=page_threshold,
            within_page_weight=within_page_weight,
            cross_page_tau=cross_page_tau,
            train_matrix=train_matrix,
            verbose=False
        ).toarray().astype(np.float32)
        
        if verbose:
            print(f"    Session matrix: shape={C_session.shape}, nnz={np.count_nonzero(C_session):,}")
        
        # Step 3: Normalize session matrix and combine
        if verbose:
            print(f"\n  [3/3] Normalizing and combining...")
        
        # Normalize C_session to [0, 1] range
        session_max = C_session.max()
        if session_max > 0:
            C_session_norm = C_session / session_max
        else:
            C_session_norm = C_session
        
        # Auto-scale: use mean of base matrix as reference
        scale = base_mean
        
        if verbose:
            print(f"    Scale factor: {scale:.2f}")
        
        # Combine: base + alpha * scale * normalized_session
        C_combined = C_base + alpha * scale * C_session_norm
        
        if verbose:
            print(f"    Combined matrix: shape={C_combined.shape}")
            print(f"    Combined nnz: {np.count_nonzero(C_combined):,}")
            print(f"    Value range: [{C_combined.min():.2f}, {C_combined.max():.2f}]")
        
        return C_combined
    
    def _normalize_matrix(self, matrix: np.ndarray, method: str = 'l1_row') -> np.ndarray:
        """
        Normalize matrix using specified method.
        
        Args:
            matrix: Input matrix (dense)
            method: Normalization method
                - 'l1_row': Each row sums to 1
                - 'l2_row': Each row has L2 norm = 1
                - 'max': Global max = 1
                - 'max_row': Each row's max = 1
            
        Returns:
            Normalized matrix
        """
        matrix = matrix.copy()
        
        if method == 'l1_row':
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return matrix / row_sums
        
        elif method == 'l2_row':
            row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1
            return matrix / row_norms
        
        elif method == 'max':
            max_val = matrix.max()
            if max_val == 0:
                return matrix
            return matrix / max_val
        
        elif method == 'max_row':
            row_maxs = matrix.max(axis=1, keepdims=True)
            row_maxs[row_maxs == 0] = 1
            return matrix / row_maxs
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader("../../../data/train/train_ratings.csv")
    train_mat, valid_mat, valid_gt = loader.create_train_valid_split()
    print(f"\nTrain matrix shape: {train_mat.shape}")
    print(f"Train matrix nnz: {train_mat.nnz}")