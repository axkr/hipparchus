package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;


public class HS267Test {

    private static class HS267Obj extends TwiceDifferentiableFunction {
        @Override
        public int dim() {
            return 5;
        }

        @Override
        public double value(RealVector x) {
            double fx = 0.0;
            for (int i = 1; i <= 11; i++) {
                double h = 0.1 * i;
                double fi = x.getEntry(2) * FastMath.exp(-x.getEntry(0) * h)
                          - x.getEntry(3) * FastMath.exp(-x.getEntry(1) * h)
                          + 3.0 * FastMath.exp(-x.getEntry(4) * h)
                          - (FastMath.exp(-h) - 5.0 * FastMath.exp(-10.0 * h) + 3.0 * FastMath.exp(-4.0 * h));
                fx += fi * fi;
            }
            return fx;
        }

        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { return null; }
    }

    private static class HS267Ineq extends InequalityConstraint {
        HS267Ineq() {
            // 8 bounds in tutto:
            // x[0] >= 0
            // x[1] >= 0
            // x[4] >= 0
            // 15 - x[i] >= 0  for i = 0..4
            super(new ArrayRealVector(new double[8]));
        }

        @Override
        public RealVector value(RealVector x) {
            double[] c = new double[8];
            c[0] = x.getEntry(0);        // x1 >= 0
            c[1] = x.getEntry(1);        // x2 >= 0
            c[2] = x.getEntry(4);        // x5 >= 0
            c[3] = 15.0 - x.getEntry(0); // x1 <= 15
            c[4] = 15.0 - x.getEntry(1); // x2 <= 15
            c[5] = 15.0 - x.getEntry(2); // x3 <= 15
            c[6] = 15.0 - x.getEntry(3); // x4 <= 15
            c[7] = 15.0 - x.getEntry(4); // x5 <= 15
            return new ArrayRealVector(c, false);
        }

        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 5; }
    }

    @Test
    public void testHS267() {
        SQPOption sqpOption = new SQPOption();
        sqpOption.setMaxLineSearchIteration(50);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(1e-11);

        InitialGuess guess = new InitialGuess(new double[]{2.0, 2.0, 2.0, 2.0, 2.0});

        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        optimizer.setDebugPrinter(System.out::println);

        double val = 0.0;
        LagrangeSolution sol = optimizer.optimize(
            sqpOption,
            guess,
            new ObjectiveFunction(new HS267Obj()),
            new HS267Ineq()
        );

        assertEquals(val, sol.getValue(), 1e-6);
    }
}
