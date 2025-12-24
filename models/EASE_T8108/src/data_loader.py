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
    Build weighted co-occurrence matrices based on time proximity.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def build_time_weighted_matrix(
        self,
        tau: float = 30.0,
        delta_t_threshold: Optional[float] = None,
        use_train_only: bool = True,
        train_matrix: Optional[csr_matrix] = None
    ) -> csr_matrix:
        """
        Build time-weighted user-item interaction matrix.
        
        Weight function: w(Δt) = exp(-Δt / tau)
        
        Args:
            tau: Time decay parameter (seconds)
            delta_t_threshold: Hard cutoff threshold (seconds), None for no cutoff
            use_train_only: If True, only use items in train_matrix
            train_matrix: Training matrix for filtering (required if use_train_only=True)
            
        Returns:
            Weighted user-item interaction matrix
        """
        print(f"Building time-weighted matrix (tau={tau}, threshold={delta_t_threshold})...")
        
        n_users = self.data_loader.n_users
        n_items = self.data_loader.n_items
        
        # Use lil_matrix for efficient construction
        weighted_matrix = lil_matrix((n_users, n_items), dtype=np.float32)
        
        if use_train_only and train_matrix is not None:
            train_csr = train_matrix.tocsr()
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            if len(seq) < 2:
                # Single item, just mark as 1
                for item_idx, _ in seq:
                    if use_train_only and train_matrix is not None:
                        if train_csr[user_idx, item_idx] > 0:
                            weighted_matrix[user_idx, item_idx] = 1.0
                    else:
                        weighted_matrix[user_idx, item_idx] = 1.0
                continue
            
            # Calculate weights based on time proximity to neighbors
            items = [item_idx for item_idx, _ in seq]
            times = [t for _, t in seq]
            
            for i, (item_idx, t) in enumerate(seq):
                if use_train_only and train_matrix is not None:
                    if train_csr[user_idx, item_idx] == 0:
                        continue
                
                # Calculate weight based on minimum time difference to neighbors
                weight = 1.0
                
                if i > 0:
                    dt_prev = abs(t - times[i-1])
                    if delta_t_threshold is None or dt_prev <= delta_t_threshold:
                        weight = max(weight, np.exp(-dt_prev / tau))
                
                if i < len(seq) - 1:
                    dt_next = abs(times[i+1] - t)
                    if delta_t_threshold is None or dt_next <= delta_t_threshold:
                        weight = max(weight, np.exp(-dt_next / tau))
                
                weighted_matrix[user_idx, item_idx] = weight
        
        return weighted_matrix.tocsr()
    
    def build_item_cooccurrence_matrix(
        self,
        tau: float = 30.0,
        delta_t_threshold: Optional[float] = None,
        train_matrix: Optional[csr_matrix] = None
    ) -> csr_matrix:
        """
        Build item-item co-occurrence matrix with time-based weights.
        WARNING: This method is O(L^2) per user and very slow for long sequences.
        Use build_windowed_cooccurrence_matrix() instead.
        
        C_ij = sum over users { sum over pairs (i,j) { w(Δt) } }
        
        Args:
            tau: Time decay parameter
            delta_t_threshold: Hard cutoff for time difference
            train_matrix: If provided, only consider items in training set
            
        Returns:
            Item-item co-occurrence matrix
        """
        print(f"Building item co-occurrence matrix (WARNING: slow O(L^2) method)...")
        
        n_items = self.data_loader.n_items
        cooc_matrix = lil_matrix((n_items, n_items), dtype=np.float32)
        
        if train_matrix is not None:
            train_csr = train_matrix.tocsr()
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            if len(seq) < 2:
                continue
            
            # Filter to training items if needed
            if train_matrix is not None:
                seq = [(item_idx, t) for item_idx, t in seq 
                       if train_csr[user_idx, item_idx] > 0]
            
            if len(seq) < 2:
                continue
            
            # Calculate pairwise co-occurrence weights
            for i in range(len(seq)):
                item_i, time_i = seq[i]
                for j in range(i + 1, len(seq)):
                    item_j, time_j = seq[j]
                    
                    dt = abs(time_j - time_i)
                    
                    if delta_t_threshold is not None and dt > delta_t_threshold:
                        continue
                    
                    weight = np.exp(-dt / tau)
                    cooc_matrix[item_i, item_j] += weight
                    cooc_matrix[item_j, item_i] += weight
        
        return cooc_matrix.tocsr()

    def build_windowed_cooccurrence_matrix(
        self,
        window_size: int = 15,
        weight_mode: str = 'uniform',
        tau: float = 30.0,
        train_matrix: Optional[csr_matrix] = None,
        verbose: bool = True
    ) -> csr_matrix:
        """
        Build item-item co-occurrence matrix using sliding window.
        
        Only considers item pairs within the same window, significantly reducing
        computation from O(L^2) to O(L * W^2) per user.
        
        Args:
            window_size: Number of items in each window (default: 15, based on MovieLens page size)
            weight_mode: How to weight pairs within window
                - 'uniform': All pairs in window get weight 1.0
                - 'distance': Weight by 1/(1 + position_diff)
                - 'time': Weight by exp(-delta_t / tau)
            tau: Time decay parameter (only used if weight_mode='time')
            train_matrix: If provided, only consider items in training set
            verbose: Print progress
            
        Returns:
            Item-item co-occurrence matrix
        """
        if verbose:
            print(f"Building windowed co-occurrence matrix...")
            print(f"  Window size: {window_size}")
            print(f"  Weight mode: {weight_mode}")
        
        n_items = self.data_loader.n_items
        n_users = len(self.data_loader.user_sequences)
        
        # Use dictionary for sparse construction (faster than lil_matrix for this pattern)
        cooc_dict = defaultdict(float)
        
        if train_matrix is not None:
            train_csr = train_matrix.tocsr()
        
        total_pairs = 0
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            if len(seq) < 2:
                continue
            
            # Filter to training items if needed
            if train_matrix is not None:
                seq = [(item_idx, t) for item_idx, t in seq 
                       if train_csr[user_idx, item_idx] > 0]
            
            if len(seq) < 2:
                continue
            
            # Sliding window approach
            seq_len = len(seq)
            
            for window_start in range(seq_len):
                window_end = min(window_start + window_size, seq_len)
                window = seq[window_start:window_end]
                
                if len(window) < 2:
                    continue
                
                # Calculate pairs within this window
                for i in range(len(window)):
                    item_i, time_i = window[i]
                    
                    for j in range(i + 1, len(window)):
                        item_j, time_j = window[j]
                        
                        # Calculate weight based on mode
                        if weight_mode == 'uniform':
                            weight = 1.0
                        elif weight_mode == 'distance':
                            pos_diff = j - i
                            weight = 1.0 / (1.0 + pos_diff)
                        elif weight_mode == 'time':
                            dt = abs(time_j - time_i)
                            weight = np.exp(-dt / tau)
                        else:
                            weight = 1.0
                        
                        # Symmetric update
                        pair_key = (min(item_i, item_j), max(item_i, item_j))
                        cooc_dict[pair_key] += weight
                        total_pairs += 1
        
        if verbose:
            print(f"  Total pairs computed: {total_pairs:,}")
            print(f"  Unique item pairs: {len(cooc_dict):,}")
        
        # Convert to sparse matrix
        if verbose:
            print(f"  Converting to sparse matrix...")
        
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

    def build_windowed_cooccurrence_matrix_fast(
        self,
        window_size: int = 15,
        train_matrix: Optional[csr_matrix] = None,
        verbose: bool = True
    ) -> csr_matrix:
        """
        Optimized version: uniform weights only, using numpy operations.
        
        Args:
            window_size: Number of items in each window
            train_matrix: If provided, only consider items in training set
            verbose: Print progress
            
        Returns:
            Item-item co-occurrence matrix (uniform weights)
        """
        if verbose:
            print(f"Building windowed co-occurrence matrix (fast mode)...")
            print(f"  Window size: {window_size}")
        
        n_items = self.data_loader.n_items
        
        # Use dictionary for sparse construction
        cooc_dict = defaultdict(int)
        
        if train_matrix is not None:
            train_csr = train_matrix.tocsr()
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            # Extract item indices only
            if train_matrix is not None:
                items = [item_idx for item_idx, t in seq 
                        if train_csr[user_idx, item_idx] > 0]
            else:
                items = [item_idx for item_idx, t in seq]
            
            if len(items) < 2:
                continue
            
            # Sliding window with numpy
            items_arr = np.array(items)
            seq_len = len(items_arr)
            
            for start in range(seq_len):
                end = min(start + window_size, seq_len)
                window = items_arr[start:end]
                
                if len(window) < 2:
                    continue
                
                # Generate all pairs in window
                for i in range(len(window)):
                    for j in range(i + 1, len(window)):
                        pair = (min(window[i], window[j]), max(window[i], window[j]))
                        cooc_dict[pair] += 1
        
        if verbose:
            print(f"  Unique item pairs: {len(cooc_dict):,}")
            print(f"  Converting to sparse matrix...")
        
        # Convert to sparse matrix
        rows, cols, data = [], [], []
        for (i, j), count in cooc_dict.items():
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(count), float(count)])
        
        cooc_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_items, n_items),
            dtype=np.float32
        )
        
        if verbose:
            print(f"  Matrix nnz: {cooc_matrix.nnz:,}")
        
        return cooc_matrix


    def build_hybrid_cooccurrence_matrix(
        self,
        max_window_size: int = 50,
        max_time_diff: float = 60.0,
        weight_mode: str = 'exponential',
        tau: float = 30.0,
        time_threshold: float = 10.0,
        session_threshold: Optional[float] = None,
        stride: int = 1,
        train_matrix: Optional[csr_matrix] = None,
        verbose: bool = True
    ) -> csr_matrix:
        """
        Build item-item co-occurrence matrix using hybrid window approach.
        
        Combines count-based and time-based window constraints for flexible
        co-occurrence computation.
        
        Window condition: (position_diff <= max_window_size) AND (time_diff <= max_time_diff)
        
        Args:
            max_window_size: Maximum window size by count (set large like 1000 for time-only mode)
            max_time_diff: Maximum time difference in seconds (set large like 1e9 for count-only mode)
            weight_mode: How to weight pairs
                - 'exponential': exp(-Δt / tau)
                - 'binary': 1 if Δt <= time_threshold else 0
            tau: Time decay parameter for exponential mode (seconds)
            time_threshold: Threshold for binary mode (seconds)
            session_threshold: If set, pairs across sessions (gap > threshold) are excluded
            stride: Window sliding step (default: 1)
            train_matrix: If provided, only consider items in training set
            verbose: Print progress
            
        Returns:
            Item-item co-occurrence matrix
        """
        if verbose:
            print(f"Building hybrid co-occurrence matrix...")
            print(f"  Max window size: {max_window_size}")
            print(f"  Max time diff: {max_time_diff}s")
            print(f"  Weight mode: {weight_mode}")
            if weight_mode == 'exponential':
                print(f"  Tau: {tau}s")
            elif weight_mode == 'binary':
                print(f"  Time threshold: {time_threshold}s")
            if session_threshold:
                print(f"  Session threshold: {session_threshold}s")
            print(f"  Stride: {stride}")
        
        n_items = self.data_loader.n_items
        
        # Use dictionary for sparse construction
        cooc_dict = defaultdict(float)
        
        if train_matrix is not None:
            train_csr = train_matrix.tocsr()
        
        total_pairs = 0
        skipped_session = 0
        skipped_time = 0
        
        for user_idx, seq in self.data_loader.user_sequences.items():
            if len(seq) < 2:
                continue
            
            # Filter to training items if needed
            if train_matrix is not None:
                seq = [(item_idx, t) for item_idx, t in seq 
                       if train_csr[user_idx, item_idx] > 0]
            
            if len(seq) < 2:
                continue
            
            seq_len = len(seq)
            
            # Sliding window with stride
            for window_start in range(0, seq_len, stride):
                # Determine window end based on max_window_size
                window_end = min(window_start + max_window_size, seq_len)
                
                # Get base item (anchor)
                base_item, base_time = seq[window_start]
                
                # Iterate through potential pairs in window
                for j in range(window_start + 1, window_end):
                    other_item, other_time = seq[j]
                    
                    # Calculate time difference
                    dt = abs(other_time - base_time)
                    
                    # Session check: skip if crossing session boundary
                    if session_threshold is not None and dt > session_threshold:
                        skipped_session += 1
                        continue
                    
                    # Time window check
                    if dt > max_time_diff:
                        skipped_time += 1
                        continue
                    
                    # Calculate weight based on mode
                    if weight_mode == 'exponential':
                        weight = np.exp(-dt / tau)
                    elif weight_mode == 'binary':
                        weight = 1.0 if dt <= time_threshold else 0.0
                    else:
                        weight = 1.0
                    
                    if weight > 0:
                        # Symmetric update (bidirectional)
                        pair_key = (min(base_item, other_item), max(base_item, other_item))
                        cooc_dict[pair_key] += weight
                        total_pairs += 1
        
        if verbose:
            print(f"  Total pairs computed: {total_pairs:,}")
            print(f"  Unique item pairs: {len(cooc_dict):,}")
            if session_threshold:
                print(f"  Pairs skipped (session): {skipped_session:,}")
            print(f"  Pairs skipped (time window): {skipped_time:,}")
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
        scale: float = None,
        max_time_diff: float = 60.0,
        weight_mode: str = 'exponential',
        tau: float = 10.0,
        time_threshold: float = 5.0,
        max_window_size: int = 50,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Build combined co-occurrence matrix using Additive approach.
        
        C_final = X^T X + alpha * scale * normalize(C_time)
        
        This preserves the base matrix (X^T X) completely and adds
        normalized time-weighted information on top.
        
        Args:
            train_matrix: User-item interaction matrix
            alpha: Weight for time-based matrix (0 = base only)
            scale: Scale factor for time matrix (None = auto, uses mean of X^T X)
            max_time_diff: Maximum time difference for time-weighted matrix
            weight_mode: 'exponential' or 'binary'
            tau: Time decay parameter for exponential mode
            time_threshold: Threshold for binary mode
            max_window_size: Maximum window size
            verbose: Print progress
            
        Returns:
            Combined co-occurrence matrix (dense, for EASE)
        """
        if verbose:
            print(f"Building combined co-occurrence matrix (Additive)...")
            print(f"  Alpha (time weight): {alpha}")
        
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
        
        # Step 2: Build time-weighted co-occurrence matrix
        if verbose:
            print(f"\n  [2/3] Computing time-weighted matrix...")
        C_time = self.build_hybrid_cooccurrence_matrix(
            max_window_size=max_window_size,
            max_time_diff=max_time_diff,
            weight_mode=weight_mode,
            tau=tau,
            time_threshold=time_threshold,
            session_threshold=None,
            stride=1,
            train_matrix=train_matrix,
            verbose=False
        ).toarray().astype(np.float32)
        
        if verbose:
            print(f"    Time matrix: shape={C_time.shape}, nnz={np.count_nonzero(C_time):,}")
        
        # Step 3: Normalize time matrix and scale
        if verbose:
            print(f"\n  [3/3] Normalizing and combining...")
        
        # Normalize C_time to [0, 1] range
        time_max = C_time.max()
        if time_max > 0:
            C_time_norm = C_time / time_max
        else:
            C_time_norm = C_time
        
        # Auto-scale: use mean of base matrix as reference
        if scale is None:
            scale = base_mean
        
        if verbose:
            print(f"    Scale factor: {scale:.2f}")
        
        # Combine: base + alpha * scale * normalized_time
        C_combined = C_base + alpha * scale * C_time_norm
        
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
            
        Returns:
            Normalized matrix
        """
        matrix = matrix.copy()
        
        if method == 'l1_row':
            # Each row sums to 1
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return matrix / row_sums
        
        elif method == 'l2_row':
            # Each row has L2 norm = 1
            row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1
            return matrix / row_norms
        
        elif method == 'max':
            # Global max = 1
            max_val = matrix.max()
            if max_val == 0:
                return matrix
            return matrix / max_val
        
        elif method == 'max_row':
            # Each row's max = 1
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