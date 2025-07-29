package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

public class HS281Test {

    private static class HS281Obj extends TwiceDifferentiableFunction {
        @Override
        public int dim() {
            return 10;
        }

        @Override
        public double value(RealVector x) {
            double sum = 0.0;
            for (int i = 0; i < 10; i++) {
                double weight = Math.pow(i + 1, 3);  // i from 0 to 9 ⇒ index from 1 to 10
                double diff = x.getEntry(i) - 1.0;
                sum += weight * diff * diff;
            }
            return Math.pow(sum, 1.0 / 3.0);
        }

        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { return null; }
    }

    @Test
    public void testHS281() {
        SQPOption sqpOption = new SQPOption();
        sqpOption.setMaxLineSearchIteration(50);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(1e-7);

        double[] start = new double[10]; // x(i) = 0
        InitialGuess guess = new InitialGuess(start);

        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        optimizer.setDebugPrinter(System.out::println);

        double val = 0.0;
        LagrangeSolution sol = optimizer.optimize(
            sqpOption,
            guess,
            new ObjectiveFunction(new HS281Obj())
        );

        assertEquals(val, sol.getValue(), 1e-4);
    }
}
