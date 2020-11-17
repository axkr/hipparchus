/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This is not the original file distributed by the Apache Software Foundation
 * It has been modified by the Hipparchus project
 */

package org.hipparchus.linear;

import org.hipparchus.UnitTestUtils;
import org.hipparchus.complex.Complex;
import org.hipparchus.exception.LocalizedCoreFormats;
import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.fraction.Fraction;
import org.hipparchus.fraction.FractionField;
import org.junit.Assert;
import org.junit.Test;

public class FieldLUDecompositionTest {
    private Fraction[][] testData = {
            { new Fraction(1), new Fraction(2), new Fraction(3)},
            { new Fraction(2), new Fraction(5), new Fraction(3)},
            { new Fraction(1), new Fraction(0), new Fraction(8)}
    };
    private Fraction[][] testDataMinus = {
            { new Fraction(-1), new Fraction(-2), new Fraction(-3)},
            { new Fraction(-2), new Fraction(-5), new Fraction(-3)},
            { new Fraction(-1),  new Fraction(0), new Fraction(-8)}
    };
    private Fraction[][] luData = {
            { new Fraction(2), new Fraction(3), new Fraction(3) },
            { new Fraction(2), new Fraction(3), new Fraction(7) },
            { new Fraction(6), new Fraction(6), new Fraction(8) }
    };

    // singular matrices
    private Fraction[][] singular = {
            { new Fraction(2), new Fraction(3) },
            { new Fraction(2), new Fraction(3) }
    };
    private Fraction[][] bigSingular = {
            { new Fraction(1), new Fraction(2),   new Fraction(3),    new Fraction(4) },
            { new Fraction(2), new Fraction(5),   new Fraction(3),    new Fraction(4) },
            { new Fraction(7), new Fraction(3), new Fraction(256), new Fraction(1930) },
            { new Fraction(3), new Fraction(7),   new Fraction(6),    new Fraction(8) }
    }; // 4th row = 1st + 2nd

    /** test dimensions */
    @Test
    public void testDimensions() {
        FieldMatrix<Fraction> matrix =
            new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldLUDecomposition<Fraction> LU = new FieldLUDecomposition<Fraction>(matrix);
        Assert.assertEquals(testData.length, LU.getL().getRowDimension());
        Assert.assertEquals(testData.length, LU.getL().getColumnDimension());
        Assert.assertEquals(testData.length, LU.getU().getRowDimension());
        Assert.assertEquals(testData.length, LU.getU().getColumnDimension());
        Assert.assertEquals(testData.length, LU.getP().getRowDimension());
        Assert.assertEquals(testData.length, LU.getP().getColumnDimension());

    }

    /** test non-square matrix */
    @Test
    public void testNonSquare() {
        try {
            // we don't use FractionField.getInstance() for testing purposes
            new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(new Fraction[][] {
                    { Fraction.ZERO, Fraction.ZERO },
                    { Fraction.ZERO, Fraction.ZERO },
                    { Fraction.ZERO, Fraction.ZERO }
            }));
            Assert.fail("Expected MathIllegalArgumentException");
        } catch (MathIllegalArgumentException ime) {
            Assert.assertEquals(LocalizedCoreFormats.NON_SQUARE_MATRIX, ime.getSpecifier());
        }
    }

    /** test PA = LU */
    @Test
    public void testPAEqualLU() {
        FieldMatrix<Fraction> matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldLUDecomposition<Fraction> lu = new FieldLUDecomposition<Fraction>(matrix);
        FieldMatrix<Fraction> l = lu.getL();
        FieldMatrix<Fraction> u = lu.getU();
        FieldMatrix<Fraction> p = lu.getP();
        UnitTestUtils.assertEquals(p.multiply(matrix), l.multiply(u));

        matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testDataMinus);
        lu = new FieldLUDecomposition<Fraction>(matrix);
        l = lu.getL();
        u = lu.getU();
        p = lu.getP();
        UnitTestUtils.assertEquals(p.multiply(matrix), l.multiply(u));

        matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), 17, 17);
        for (int i = 0; i < matrix.getRowDimension(); ++i) {
            matrix.setEntry(i, i, Fraction.ONE);
        }
        lu = new FieldLUDecomposition<Fraction>(matrix);
        l = lu.getL();
        u = lu.getU();
        p = lu.getP();
        UnitTestUtils.assertEquals(p.multiply(matrix), l.multiply(u));

        matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), singular);
        lu = new FieldLUDecomposition<Fraction>(matrix);
        Assert.assertFalse(lu.getSolver().isNonSingular());
        Assert.assertNull(lu.getL());
        Assert.assertNull(lu.getU());
        Assert.assertNull(lu.getP());

        matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), bigSingular);
        lu = new FieldLUDecomposition<Fraction>(matrix);
        Assert.assertFalse(lu.getSolver().isNonSingular());
        Assert.assertNull(lu.getL());
        Assert.assertNull(lu.getU());
        Assert.assertNull(lu.getP());

    }

    /** test that L is lower triangular with unit diagonal */
    @Test
    public void testLLowerTriangular() {
        FieldMatrix<Fraction> matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldMatrix<Fraction> l = new FieldLUDecomposition<Fraction>(matrix).getL();
        for (int i = 0; i < l.getRowDimension(); i++) {
            Assert.assertEquals(Fraction.ONE, l.getEntry(i, i));
            for (int j = i + 1; j < l.getColumnDimension(); j++) {
                Assert.assertEquals(Fraction.ZERO, l.getEntry(i, j));
            }
        }
    }

    /** test that U is upper triangular */
    @Test
    public void testUUpperTriangular() {
        FieldMatrix<Fraction> matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldMatrix<Fraction> u = new FieldLUDecomposition<Fraction>(matrix).getU();
        for (int i = 0; i < u.getRowDimension(); i++) {
            for (int j = 0; j < i; j++) {
                Assert.assertEquals(Fraction.ZERO, u.getEntry(i, j));
            }
        }
    }

    /** test that P is a permutation matrix */
    @Test
    public void testPPermutation() {
        FieldMatrix<Fraction> matrix = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldMatrix<Fraction> p   = new FieldLUDecomposition<Fraction>(matrix).getP();

        FieldMatrix<Fraction> ppT = p.multiply(p.transpose());
        FieldMatrix<Fraction> id  =
            new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(),
                                          p.getRowDimension(), p.getRowDimension());
        for (int i = 0; i < id.getRowDimension(); ++i) {
            id.setEntry(i, i, Fraction.ONE);
        }
        UnitTestUtils.assertEquals(id, ppT);

        for (int i = 0; i < p.getRowDimension(); i++) {
            int zeroCount  = 0;
            int oneCount   = 0;
            int otherCount = 0;
            for (int j = 0; j < p.getColumnDimension(); j++) {
                final Fraction e = p.getEntry(i, j);
                if (e.equals(Fraction.ZERO)) {
                    ++zeroCount;
                } else if (e.equals(Fraction.ONE)) {
                    ++oneCount;
                } else {
                    ++otherCount;
                }
            }
            Assert.assertEquals(p.getColumnDimension() - 1, zeroCount);
            Assert.assertEquals(1, oneCount);
            Assert.assertEquals(0, otherCount);
        }

        for (int j = 0; j < p.getColumnDimension(); j++) {
            int zeroCount  = 0;
            int oneCount   = 0;
            int otherCount = 0;
            for (int i = 0; i < p.getRowDimension(); i++) {
                final Fraction e = p.getEntry(i, j);
                if (e.equals(Fraction.ZERO)) {
                    ++zeroCount;
                } else if (e.equals(Fraction.ONE)) {
                    ++oneCount;
                } else {
                    ++otherCount;
                }
            }
            Assert.assertEquals(p.getRowDimension() - 1, zeroCount);
            Assert.assertEquals(1, oneCount);
            Assert.assertEquals(0, otherCount);
        }

    }


    /** test singular */
    @Test
    public void testSingular() {
        final FieldMatrix<Fraction> m = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData);
        FieldLUDecomposition<Fraction> lu = new FieldLUDecomposition<Fraction>(m);
        Assert.assertTrue(lu.getSolver().isNonSingular());
        Assert.assertEquals(new Fraction(-1, 1), lu.getDeterminant());
        lu = new FieldLUDecomposition<>(m.getSubMatrix(0, 1, 0, 1));
        Assert.assertTrue(lu.getSolver().isNonSingular());
        Assert.assertEquals(new Fraction(+1, 1), lu.getDeterminant());
        lu = new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), singular));
        Assert.assertFalse(lu.getSolver().isNonSingular());
        Assert.assertEquals(new Fraction(0, 1), lu.getDeterminant());
        lu = new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), bigSingular));
        Assert.assertFalse(lu.getSolver().isNonSingular());
        Assert.assertEquals(new Fraction(0, 1), lu.getDeterminant());
        try {
            lu.getSolver().solve(new ArrayFieldVector<>(new Fraction[] { Fraction.ONE, Fraction.ONE, Fraction.ONE, Fraction.ONE }));
            Assert.fail("an exception should have been thrown");
        } catch (MathIllegalArgumentException miae) {
            Assert.assertEquals(LocalizedCoreFormats.SINGULAR_MATRIX, miae.getSpecifier());
        }
    }

    /** test matrices values */
    @Test
    public void testMatricesValues1() {
       FieldLUDecomposition<Fraction> lu =
            new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData));
        FieldMatrix<Fraction> lRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(1), new Fraction(0), new Fraction(0) },
                { new Fraction(2), new Fraction(1), new Fraction(0) },
                { new Fraction(1), new Fraction(-2), new Fraction(1) }
        });
        FieldMatrix<Fraction> uRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(1),  new Fraction(2), new Fraction(3) },
                { new Fraction(0), new Fraction(1), new Fraction(-3) },
                { new Fraction(0),  new Fraction(0), new Fraction(-1) }
        });
        FieldMatrix<Fraction> pRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(1), new Fraction(0), new Fraction(0) },
                { new Fraction(0), new Fraction(1), new Fraction(0) },
                { new Fraction(0), new Fraction(0), new Fraction(1) }
        });
        int[] pivotRef = { 0, 1, 2 };

        // check values against known references
        FieldMatrix<Fraction> l = lu.getL();
        UnitTestUtils.assertEquals(lRef, l);
        FieldMatrix<Fraction> u = lu.getU();
        UnitTestUtils.assertEquals(uRef, u);
        FieldMatrix<Fraction> p = lu.getP();
        UnitTestUtils.assertEquals(pRef, p);
        int[] pivot = lu.getPivot();
        for (int i = 0; i < pivotRef.length; ++i) {
            Assert.assertEquals(pivotRef[i], pivot[i]);
        }

        // check the same cached instance is returned the second time
        Assert.assertTrue(l == lu.getL());
        Assert.assertTrue(u == lu.getU());
        Assert.assertTrue(p == lu.getP());

    }

    /** test matrices values */
    @Test
    public void testMatricesValues2() {
       FieldLUDecomposition<Fraction> lu =
            new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), luData));
        FieldMatrix<Fraction> lRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(1), new Fraction(0), new Fraction(0) },
                { new Fraction(3), new Fraction(1), new Fraction(0) },
                { new Fraction(1), new Fraction(0), new Fraction(1) }
        });
        FieldMatrix<Fraction> uRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(2), new Fraction(3), new Fraction(3)    },
                { new Fraction(0), new Fraction(-3), new Fraction(-1)  },
                { new Fraction(0), new Fraction(0), new Fraction(4) }
        });
        FieldMatrix<Fraction> pRef = new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), new Fraction[][] {
                { new Fraction(1), new Fraction(0), new Fraction(0) },
                { new Fraction(0), new Fraction(0), new Fraction(1) },
                { new Fraction(0), new Fraction(1), new Fraction(0) }
        });
        int[] pivotRef = { 0, 2, 1 };

        // check values against known references
        FieldMatrix<Fraction> l = lu.getL();
        UnitTestUtils.assertEquals(lRef, l);
        FieldMatrix<Fraction> u = lu.getU();
        UnitTestUtils.assertEquals(uRef, u);
        FieldMatrix<Fraction> p = lu.getP();
        UnitTestUtils.assertEquals(pRef, p);
        int[] pivot = lu.getPivot();
        for (int i = 0; i < pivotRef.length; ++i) {
            Assert.assertEquals(pivotRef[i], pivot[i]);
        }

        // check the same cached instance is returned the second time
        Assert.assertTrue(l == lu.getL());
        Assert.assertTrue(u == lu.getU());
        Assert.assertTrue(p == lu.getP());
    }

    @Test
    public void testSignedZeroPivot() {
        FieldMatrix<Complex> m = new Array2DRowFieldMatrix<>(new Complex[][] {
            { new Complex(-0.0, 0.0), Complex.ONE },
            { Complex.ONE, Complex.ZERO }
        });
        FieldVector<Complex> v = new ArrayFieldVector<>(new Complex[] {
            new Complex(2, 0),
            new Complex(0, 2)
        });
        FieldDecompositionSolver<Complex> solver = new FieldLUDecomposition<>(m).getSolver();
        FieldVector<Complex> u = solver.solve(v);
        Assert.assertEquals(u.getEntry(0), new Complex(0, 2));
        Assert.assertEquals(u.getEntry(1), new Complex(2, 0));
    }

    @Test
    public void testsolve() {
        FieldLUDecomposition<Fraction> lu =
                        new FieldLUDecomposition<Fraction>(new Array2DRowFieldMatrix<Fraction>(FractionField.getInstance(), testData));
        FieldDecompositionSolver<Fraction> solver = lu.getSolver();
        FieldVector<Fraction> solution = solver.solve(new ArrayFieldVector<>(new Fraction[] {
            new Fraction(1, 2), new Fraction(2, 3), new Fraction(3,4)
        }));
        Assert.assertEquals(testData.length, solution.getDimension());
        Assert.assertEquals(new Fraction(-31, 12), solution.getEntry(0));
        Assert.assertEquals(new Fraction( 11, 12), solution.getEntry(1));
        Assert.assertEquals(new Fraction(  5, 12), solution.getEntry(2));
        try {
            solver.solve(new ArrayFieldVector<>(new Fraction[] { Fraction.ONE, Fraction.ONE }));
            Assert.fail("an exception should have been thrown");
        } catch (MathIllegalArgumentException miae) {
            Assert.assertEquals(LocalizedCoreFormats.DIMENSIONS_MISMATCH, miae.getSpecifier());
        }
    }

}
