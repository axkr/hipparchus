package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.optim.nonlinear.vector.constrained.LagrangeSolution;
import org.hipparchus.optim.nonlinear.vector.constrained.EqualityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.InequalityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.TwiceDifferentiableFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

public class HS052Test {
    private static final double pi = FastMath.PI;

    private static class HS052Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 5; }
        @Override public double value(RealVector x) {
            return (((FastMath.pow(((4.0 * x.getEntry(0)) - x.getEntry(1)), 2) + FastMath.pow(((x.getEntry(1) + x.getEntry(2)) - 2.0), 2)) + FastMath.pow((x.getEntry(3) - 1.0), 2)) + FastMath.pow((x.getEntry(4) - 1.0), 2));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS052Eq extends EqualityConstraint {
        HS052Eq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ ((x.getEntry(0) + (3.0 * x.getEntry(1)))) - (0.0), (((x.getEntry(2) + x.getEntry(3)) - (2.0 * x.getEntry(4)))) - (0.0), ((x.getEntry(1) - x.getEntry(4))) - (0.0) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 5; }
    }

    @Test
    public void testHS052() {
        InitialGuess guess = new InitialGuess(new double[]{2.0, 2.0, 2.0, 2.0, 2.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = (1859.0 / 349.0);
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS052Obj()), new HS052Eq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}