## Approach

I use Acklam's rational approximation with literature coefficients. Acklam's method uses two regions (central and tails) with different rational approximations for each. I add Halley refinement for full double precision and optional SIMD/OpenMP vectorization.

## Piecewise Forms

I use Acklam's published rational approximations with two regions:

**Central region (0.02425 ≤ x ≤ 0.97575):** Exploits symmetry around x = 0.5 with u = x - 0.5, r = u², then Φ⁻¹(x) ≈ u · A(r)/B(r) where A and B are degree-5 polynomials. Using u·(rational in u²) gives automatic symmetry.

**Tail region (x < 0.02425 or x > 0.97575):** Uses change of variable t = √(-2 log m) where m = min(x, 1-x). Then Φ⁻¹(x) ≈ ±C(t)/D(t) where C is degree-5 and D is degree-4. This transform makes the infinite tail range tractable for polynomial approximation.

Both regions use Horner's method for numerical stability. Join at x_low = 0.02425 ≈ Φ(-2).

## Halley Refinement

One Halley iteration after the rational approximation achieves full precision:

    z_new = z - r / (1 - 1/2zr)  where r = (Φ(z) - x) / φ(z)

The assignment notes that direct computation of r causes catastrophic cancellation in extreme tails. Following the suggested approach, I use logarithmic forms with expm1:

Right tail (x > 1 - 10⁻⁵): r = -(1-x) · expm1(log Φ(-z) - log(1-x)) / φ(z)
Left tail (x < 10⁻⁵): r = x · expm1(log Φ(z) - log x) / φ(z)
Central region: r = (Φ(z) - x) / φ(z)

This avoids subtractive cancellation and maintains accuracy near 0 and 1.

## Error Evaluation

I validated the implementation by comparing against SciPy norm.ppf on 1 million linearly-spaced points in [10⁻¹², 1-10⁻¹²].

Results:

- Raw Acklam approximation: max error 7.36×10⁻⁹, mean error ≈ 7×10⁻¹²
- With Halley refinement: max error 8.58×10⁻¹³, mean error ≈ 2×10⁻¹⁶
- Bisection baseline: max error 7.74×10⁻⁶

Property validation on 10,000 test points:

- Symmetry Φ⁻¹(1-x) = -Φ⁻¹(x): max error 2.26×10⁻¹⁴
- Monotonicity: zero violations (strictly increasing)
- Round-trip Φ(Φ⁻¹(x)) ≈ x: max relative error 2.98×10⁻¹⁵
- Derivative d/dx Φ⁻¹(x) = 1/φ(Φ⁻¹(x)): max relative error 4.59×10⁻⁶
- Edge cases x=0 and x=1 correctly return ±∞

## Performance

Test machine: AMD Ryzen 9 3900X (24 cores), 1 million evaluations

| Configuration                   | Time (s) | Throughput | Speedup | Max Error  | Mean Error |
| ------------------------------- | -------- | ---------- | ------- | ---------- | ---------- |
| Bisection baseline              | 1.1371   | 0.9 M/s    | 1.0×    | 7.74×10⁻⁶  | 7.74×10⁻¹² |
| Acklam scalar                   | 0.0259   | 38.5 M/s   | 44×     | 8.58×10⁻¹³ | 2.02×10⁻¹⁶ |
| Acklam + SIMD                   | 0.0215   | 46.6 M/s   | 53×     | 8.57×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + OpenMP                 | 0.0052   | 193.3 M/s  | 220×    | 8.58×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + SIMD + OpenMP          | 0.0035   | 285.5 M/s  | 325×    | 8.57×10⁻¹³ | 2.01×10⁻¹⁶ |
| Acklam + SIMD + OpenMP (no ref) | 0.0015   | 649.8 M/s  | 739×    | 7.36×10⁻⁹  | 5.71×10⁻¹⁰ |

## Vectorization

- SIMD: AVX intrinsics (256-bit, 4 doubles) with FMA for polynomial evaluation
- OpenMP: Parallel loop over SIMD chunks
- Central region computed in SIMD, tails fall back to scalar due to branching
- Auto-detection via **AVX** macro, fallback to scalar+OpenMP if unavailable

## Non-idealities

- Literature coefficients rather than custom-fitted, but thoroughly validated
- Tail handling requires scalar fallback (minimal impact, tails are rare)
- x ≤ 0 or x ≥ 1 return ±∞ per IEEE convention

Reference: Peter J. Acklam, "An algorithm for computing the inverse normal cumulative distribution function" (2010)  
https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
