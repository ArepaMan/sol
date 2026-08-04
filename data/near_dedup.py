"""MinHash + LSH-banding near-duplicate detection, on word 5-gram shingles.

Off by default (`--near-dedup` in prepare.py) because it is O(n) with a real
constant factor and, on the full ~2.1M-story TinyStories train split, would
take considerably longer than exact dedup for a corpus that turns out to be
mostly exact-duplicate-free. Exercised for correctness in tests on small
synthetic corpora; run for real only if the exact-dup rate logged by
prepare.py suggests it is worth the wait.

Each of the `num_perm` hash functions is BLAKE2B salted with its own index
rather than the classic (a*x + b mod p) universal-hash family — that formula
needs the intermediate product to fit the working integer type, and a naive
numpy int64 implementation overflows the moment a and the shingle hash are
both drawn from a ~61-bit range. Python ints are arbitrary-precision, so
salted hashing sidesteps the overflow class of bug entirely at a modest
constant-factor cost, which near-dedup already pays for being O(n) at all.
"""

from __future__ import annotations

import hashlib


def _shingles(text: str, k: int = 5) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _hash_shingle(shingle: str, salt: int) -> int:
    digest = hashlib.blake2b(f"{salt}:{shingle}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


_SENTINEL = (1 << 64) - 1  # larger than any real digest; marks "no shingles"


class MinHasher:
    def __init__(self, num_perm: int = 32, seed: int = 42):
        self.num_perm = num_perm
        self.salts = list(range(seed, seed + num_perm))

    def signature(self, shingles: list[str]) -> tuple[int, ...]:
        if not shingles:
            return (_SENTINEL,) * self.num_perm
        return tuple(min(_hash_shingle(s, salt) for s in shingles) for salt in self.salts)


def find_near_duplicates(
    texts: list[str],
    k: int = 5,
    num_perm: int = 32,
    bands: int = 16,
    threshold: float = 0.8,
    seed: int = 42,
) -> set[int]:
    """Returns indices of texts to drop — later near-duplicates of an earlier text.

    LSH banding (b bands x r rows, num_perm = b*r) buckets documents that agree
    on an entire band, so only genuine candidates pay for the exact signature
    comparison, rather than every pair.
    """
    if num_perm % bands != 0:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
    rows = num_perm // bands

    hasher = MinHasher(num_perm=num_perm, seed=seed)
    signatures = [hasher.signature(_shingles(t, k)) for t in texts]

    buckets: dict[tuple, list[int]] = {}
    to_drop: set[int] = set()

    for i, sig in enumerate(signatures):
        if i in to_drop:
            continue
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows])
            bucket = buckets.setdefault(key, [])
            candidates.update(bucket)
            bucket.append(i)
        for j in candidates:
            if j in to_drop:
                continue
            est_jaccard = sum(a == b for a, b in zip(sig, signatures[j])) / num_perm
            if est_jaccard >= threshold:
                to_drop.add(i)
                break

    return to_drop
