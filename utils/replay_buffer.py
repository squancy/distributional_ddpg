import numpy as np


class ReplayAlphaStratified:
    """
    Replay buffer partitioned into some number of bins of sub-buffers by alpha value.
    Ensures all risk levels appear in every training batch.

    Attributes:
        batch_size (int): Batch size.
        n_bins (int): Number of bins.
        bin_size (int): Size of each bin (size of memory divided by the number of bins).
        dtype (np.dtype): Data type of each bin.
        states (list[np.array]): Per-bin states.
        actions (list[np.array]): Per-bin actions.
        next_states (list[np.array]): Per-bin next states.
        rewards (list[np.array]): Per-bin rewards.
        risks (list[np.array]): Per-bin risks (scaled Markowitz portfolio return).
        alphas (list[np.array]): Per-bin alpha values.
        terminals (list[np.array]): Per-bin booleans indicating whether an episode has ended.
        pos (list[int]): First non-empty position in each bin.
        full (list[bool]): List of booleans indicating whether each bin is full or not.
        min_alpha (float = 0.05): Minimum alpha value.
        max_alpha (float = 0.5): Maximum alpha value.
    """

    def __init__(
        self,
        memory_size: int,
        batch_size: int,
        n_bins: int = 10,
        dtype: np.dtype = np.float32,
        min_alpha: float = 0.05,
        max_alpha: float = 0.5,
    ):
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.bin_size = memory_size // n_bins
        self.dtype = dtype
        self.states = [None] * n_bins
        self.actions = [None] * n_bins
        self.next_states = [None] * n_bins
        self.rewards = [np.empty(self.bin_size) for _ in range(n_bins)]
        self.risks = [np.empty(self.bin_size) for _ in range(n_bins)]
        self.alphas = [np.empty(self.bin_size) for _ in range(n_bins)]
        self.terminals = [np.empty(self.bin_size, dtype=np.int8) for _ in range(n_bins)]
        self.pos = [0] * n_bins
        self.full = [False] * n_bins
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def _bin(self, alpha: float) -> int:
        """
        Given alpha between `self.min_alpha` and `self.max_alpha`,
        it calculates the index of the bin to be used.

        Args:
            alpha (float): Alpha value.

        Returns:
            int: Index of bin.
        """
        return min(int(alpha * self.n_bins), self.n_bins - 1)

    def feed(self, experience: tuple):
        """
        Add the collected experience to the state.

        Args:
            experience (tuple): Tuple containing the state,
                action, reward, risk, next state, alpha and done.
        """
        state, action, reward, risk, next_state, alpha, done = experience
        b = self._bin(float(alpha))
        p = self.pos[b]
        if self.states[b] is None:
            self.states[b] = np.empty((self.bin_size,) + state.shape, dtype=self.dtype)
            self.actions[b] = np.empty((self.bin_size,) + action.shape, dtype=self.dtype)
            self.next_states[b] = np.empty((self.bin_size,) + next_state.shape, dtype=self.dtype)
        self.states[b][p] = state
        self.actions[b][p] = action
        self.rewards[b][p] = reward
        self.risks[b][p] = risk
        self.next_states[b][p] = next_state
        self.alphas[b][p] = alpha
        self.terminals[b][p] = done
        self.pos[b] += 1
        if self.pos[b] == self.bin_size:
            self.full[b] = True
            self.pos[b] = 0

    def size(self) -> int:
        """
        Returns the size of the replay buffer.

        Returns:
            int: Size of replay buffer.
        """
        return sum(self.bin_size if self.full[b] else self.pos[b] for b in range(self.n_bins))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer, evenly distributed
        across bins as much as possible.

        Returns:
            list: Batch of experiences.
        """
        non_empty = [b for b in range(self.n_bins) if self.full[b] or self.pos[b] > 0]
        n = len(non_empty)
        base, rem = divmod(self.batch_size, n)
        counts = [base + (1 if i < rem else 0) for i in range(n)]
        s, a, r, rk, ns, al, t = [], [], [], [], [], [], []
        for b, cnt in zip(non_empty, counts):
            ub = self.bin_size if self.full[b] else self.pos[b]
            # noqa: NPY002 — global legacy RNG kept on purpose for stream-
            # equivalence with the reference implementation; do not modernize.
            idx = np.random.randint(0, ub, size=cnt)  # noqa: NPY002
            s.append(self.states[b][idx])
            a.append(self.actions[b][idx])
            r.append(self.rewards[b][idx])
            rk.append(self.risks[b][idx])
            ns.append(self.next_states[b][idx])
            al.append(self.alphas[b][idx])
            t.append(self.terminals[b][idx])
        return [
            np.concatenate(s),
            np.concatenate(a),
            np.concatenate(r),
            np.concatenate(rk),
            np.concatenate(ns),
            np.concatenate(al),
            np.concatenate(t),
        ]


class ReplayMemory:
    """
    Simple fixed-capacity circular replay buffer.

    Used by the fixed-alpha variant, where alpha is constant within a run, so
    there is nothing to stratify over.

    Attributes:
        memory_size (int): Capacity of the buffer.
        batch_size (int): Batch size drawn by `sample`.
        dtype (np.dtype): Data type of states/actions/next_states.
        states (np.array | None): Stored states (lazily allocated on first feed).
        actions (np.array | None): Stored actions (lazily allocated).
        next_states (np.array | None): Stored next states (lazily allocated).
        rewards (np.array): Stored rewards.
        terminals (np.array): Stored done flags.
        pos (int): Next write position (wraps around).
    """

    def __init__(self, memory_size: int, batch_size: int, dtype: np.dtype = np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.states = None  # lazy-allocated on first feed
        self.actions = None
        self.next_states = None
        self.rewards = np.empty(memory_size)
        self.terminals = np.empty(memory_size, dtype=np.int8)
        self.pos = 0
        self._size = 0

    def feed(self, experience: tuple):
        """
        Add a single transition to the buffer, overwriting the oldest once full.

        Args:
            experience (tuple): (state, action, reward, next_state, done).
        """
        state, action, reward, next_state, done = experience
        p = self.pos
        if self.states is None:
            self.states = np.empty((self.memory_size,) + state.shape, dtype=self.dtype)
            self.actions = np.empty((self.memory_size,) + action.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size,) + state.shape, dtype=self.dtype)
        self.states[p] = state
        self.actions[p] = action
        self.rewards[p] = reward
        self.next_states[p] = next_state
        self.terminals[p] = done
        self.pos = (p + 1) % self.memory_size
        self._size = min(self._size + 1, self.memory_size)

    def size(self) -> int:
        """
        Returns the number of transitions currently stored.

        Returns:
            int: Current buffer size.
        """
        return self._size

    def sample(self) -> list:
        """
        Sample a uniform random batch of transitions.

        Returns:
            list: [states, actions, rewards, next_states, terminals].
        """
        idx = np.random.randint(0, self._size, size=self.batch_size)  # noqa: NPY002
        return [
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminals[idx],
        ]
