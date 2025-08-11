package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;


public class HS299Test {

    private static class HS299Obj extends TwiceDifferentiableFunction {

        @Override
        public int dim() {
            return 100;
        }

        @Override
        public double value(RealVector x) {
            int n = dim();
            double fx = 0.0;
            for (int i = 0; i < n - 1; i++) {
                double xi = x.getEntry(i);
                double xip1 = x.getEntry(i + 1);
                double fi = 10.0 * (xip1 - xi * xi);
                double gi = 1.0 - xi;
                fx += fi * fi + gi * gi;
            }
            return fx * 1.0e-4;
        }

        @Override
        public RealVector gradient(RealVector x) {
            throw new UnsupportedOperationException();
        }

        @Override
        public RealMatrix hessian(RealVector x) {
            return null;
        }
    }

    @Test
    public void testHS299() {
        SQPOption sqpOption = new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(1e-11);

        // Punto iniziale come da TP299: x(i) = -1.2, ma x(2i) = 1.0
        double[] start = new double[100];
        for (int i = 0; i < 100; i++) {
            start[i] = -1.2;
        }
        for (int i = 1; i < 100; i += 2) {
            start[i] = 1.0;
        }

        InitialGuess guess = new InitialGuess(start);
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        optimizer.setDebugPrinter(System.out::println);

        double val = 0.0;
        LagrangeSolution sol = optimizer.optimize(sqpOption, guess, new ObjectiveFunction(new HS299Obj()));

        assertEquals(val, sol.getValue(), 1e-6);
    }
}
