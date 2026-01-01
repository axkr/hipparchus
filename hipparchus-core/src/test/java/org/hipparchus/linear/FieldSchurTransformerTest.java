package org.hipparchus.linear;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import java.util.Arrays;
import org.hipparchus.complex.Complex;
import org.hipparchus.complex.ComplexField;
import org.junit.jupiter.api.Test;

/**
 * Test class for FieldSchurTransformer.
 */
public class FieldSchurTransformerTest {

  private final double epsilon = 1.0e-10;

  /**
   * Helper method to verify the Schur Decomposition properties. 1. Check if T is Upper Triangular.
   * 2. Check if P is Unitary (P * P^H = I). 3. Check if A = P * T * P^H.
   */
  private void checkSchurDecomposition(FieldMatrix<Complex> A) {
    FieldSchurTransformer transformer = new FieldSchurTransformer(A);
    FieldMatrix<Complex> P = transformer.getP();
    FieldMatrix<Complex> T = transformer.getT();
    FieldMatrix<Complex> PT = transformer.getPT(); // P^H (Conjugate Transpose)

    // System.out.println(P);
    // System.out.println(T);

    int n = A.getRowDimension();
    // Check T is Upper Triangular
    for (int r = 0; r < n; r++) {
      for (int c = 0; c < r; c++) {
        assertEquals(0.0, T.getEntry(r, c).norm(), epsilon,
            "T should be upper triangular. Entry (" + r + "," + c + ") is not zero.");
      }
    }

    // Check P is Unitary: P * P^H = I
    FieldMatrix<Complex> identity =
        MatrixUtils.createFieldIdentityMatrix(ComplexField.getInstance(), n);
    FieldMatrix<Complex> PPt = P.multiply(PT);

    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        Complex expected = identity.getEntry(r, c);
        Complex actual = PPt.getEntry(r, c);
        assertEquals(0.0, expected.subtract(actual).norm(), epsilon,
            "P should be unitary. P*P^H entry (" + r + "," + c + ") mismatch.");
      }
    }

    // Check A = P * T * P^H
    FieldMatrix<Complex> reconstructedA = P.multiply(T).multiply(PT);
    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        Complex expected = A.getEntry(r, c);
        Complex actual = reconstructedA.getEntry(r, c);
        assertEquals(0.0, expected.subtract(actual).norm(), epsilon,
            "Reconstruction A = P*T*P^H failed at (" + r + "," + c + ").");
      }
    }
  }

  /**
   * Example 1: Real valued matrix, but treated as Complex. SchurDecomposition[{{-4, -3}, {2, 1}}]
   * Expected Eigenvalues on diagonal of T: -1, -2
   */
  @Test
  public void testExample1() {
    // Matrix A = {{-4, -3}, {2, 1}}
    Complex[][] data =
        {{new Complex(-4, 0), new Complex(-3, 0)}, {new Complex(2, 0), new Complex(1, 0)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);

    // Additional check: The diagonal of T should contain the eigenvalues (-1 and -2).
    // Note: The order depends on the pivot strategy, so we check if they are present.
    FieldSchurTransformer transformer = new FieldSchurTransformer(matrix);
    FieldMatrix<Complex> T = transformer.getT();

    Complex t00 = T.getEntry(0, 0);
    Complex t11 = T.getEntry(1, 1);

    // Check if diagonal elements are either -1 or -2
    boolean hasMinus1 = (t00.subtract(-1).norm() < epsilon) || (t11.subtract(-1).norm() < epsilon);
    boolean hasMinus2 = (t00.subtract(-2).norm() < epsilon) || (t11.subtract(-2).norm() < epsilon);

    assertTrue(hasMinus1, "Schur form T should contain eigenvalue -1");
    assertTrue(hasMinus2, "Schur form T should contain eigenvalue -2");
  }

  /**
   * Example 2: Simple Complex Matrix (Hermitian). m = {{-1, I}, {-I, 1}} Trace = 0, Det = -2.
   * Eigenvalues should be +/- Sqrt(2).
   */
  @Test
  public void testExample2() {
    // Matrix A = {{-1, I}, {-I, 1}}
    Complex I = Complex.I;
    Complex[][] data = {{new Complex(-1), I}, //
        {I.negate(), new Complex(1, 0)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);

    // Verify eigenvalues +/- Sqrt(2)
    FieldSchurTransformer transformer = new FieldSchurTransformer(matrix);
    FieldMatrix<Complex> T = transformer.getT();

    double sqrt2 = Math.sqrt(2.0);
    Complex t00 = T.getEntry(0, 0);
    Complex t11 = T.getEntry(1, 1);

    // Since the matrix is Hermitian, T should be diagonal (Spectral Theorem)
    assertEquals(0.0, T.getEntry(0, 1).norm(), epsilon,
        "T should be diagonal for Hermitian matrix");

    // Check Eigenvalues magnitude
    assertEquals(sqrt2, t00.norm(), epsilon);
    assertEquals(sqrt2, t11.norm(), epsilon);
  }

  /**
   * Example 3: 3x3 Matrix with real entries. m = {{5, 7, 6}, {-3, 1, 6}, {-2, 1, 3}} Complex
   * eigenvalues exist.
   */
  @Test
  public void testExample3() {
    // Matrix A = {{5, 7, 6}, {-3, 1, 6}, {-2, 1, 3}}
    Complex[][] data = {{new Complex(5), new Complex(7), new Complex(6)}, //
        {new Complex(-3), new Complex(1), new Complex(6)}, //
        {new Complex(-2), new Complex(1), new Complex(3)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);
  }

  @Test
  public void testExample4() {
    // Matrix A = {{Pi, .3}, {I, 1.0+1.5}}
    Complex I = Complex.I;
    Complex[][] data = {{new Complex(Math.PI), new Complex(0.3)}, //
        {I, new Complex(1.0, 1.5)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);
  }

  @Test
  public void testExample5() {
    // Matrix A = {{-1.2, 2.7, 3.8}, {4.2, 4.4, 5.3}, {3.5, 7.6, 6.8}}
    Complex[][] data = {{new Complex(-1.2), new Complex(2.7), new Complex(3.8)}, //
        {new Complex(4.2), new Complex(4.4), new Complex(5.3)}, //
        {new Complex(3.5), new Complex(7.6), new Complex(6.8)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);
  }

  /**
   * Additional Test: Purely Imaginary Diagonal. A = {{3i, 0}, {0, -2i}} Already in Schur form
   * (Diagonal).
   */
  @Test
  public void testAlreadySchurForm() {
    Complex I = Complex.I;
    Complex[][] data = {{I.multiply(3), Complex.ZERO}, {Complex.ZERO, I.multiply(-2)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);

    FieldSchurTransformer transformer = new FieldSchurTransformer(matrix);
    FieldMatrix<Complex> T = transformer.getT();

    // Should roughly remain the same (order might flip depending on implementation stability)
    double norm0 = T.getEntry(0, 0).norm();
    double norm1 = T.getEntry(1, 1).norm();

    assertTrue((Math.abs(norm0 - 3.0) < epsilon && Math.abs(norm1 - 2.0) < epsilon)
        || (Math.abs(norm0 - 2.0) < epsilon && Math.abs(norm1 - 3.0) < epsilon));
  }

  /**
   * Additional Test: Defective Matrix (Jordan Block). A = {{1, 1}, {0, 1}} Cannot be diagonalized,
   * but Schur form exists (itself).
   */
  @Test
  public void testDefectiveMatrix() {
    Complex[][] data = {{Complex.ONE, Complex.ONE}, {Complex.ZERO, Complex.ONE}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);

    checkSchurDecomposition(matrix);

    FieldSchurTransformer transformer = new FieldSchurTransformer(matrix);
    FieldMatrix<Complex> T = transformer.getT();

    // T should still be {{1, 1}, {0, 1}} or similar
    assertEquals(1.0, T.getEntry(0, 0).getReal(), epsilon);
    assertEquals(1.0, T.getEntry(1, 1).getReal(), epsilon);
  }

  /**
   * Positive definite matrix
   */
  @Test
  public void testGetEigenvalues() {
    // Matrix A = {{5.42,3.26 + 0.643*I,-0.467-0.193*I},
    // {3.26-0.643*I,3.82,1.04-2.35*I},
    // {-0.467+0.193*I,1.04+2.35*I,4.88}}
    Complex[][] data = {{new Complex(5.42), new Complex(3.26, 0.643), new Complex(-0.467, -0.193)}, //
        {new Complex(3.26, -0.643), new Complex(3.82), new Complex(1.04, -2.35)}, //
        {new Complex(-0.467, 0.193), new Complex(1.04, 2.35), new Complex(4.88)}};
    FieldMatrix<Complex> matrix = MatrixUtils.createFieldMatrix(data);
    checkSchurDecomposition(matrix);

    Complex[] eigenvalues = FieldSchurTransformer.getEigenvalues(matrix);
    // sort descending
    Arrays.sort(eigenvalues, (a, b) -> {
      int realComp = Double.compare(b.getReal(), a.getReal());
      if (realComp != 0) {
        return realComp;
      } else {
        return Double.compare(b.getImaginary(), a.getImaginary());
      }
    });
    FieldVector<Complex> fieldVector = MatrixUtils.createFieldVector(eigenvalues);
    assertEquals(fieldVector.toString(), //
        "{(8.768464618713097, 4.605614931476993E-16); " //
            + "(5.163611813754241, -6.965822013958481E-17); " //
            + "(0.18792356753266304, -1.6208167199054114E-16)}"); //

  }
}
