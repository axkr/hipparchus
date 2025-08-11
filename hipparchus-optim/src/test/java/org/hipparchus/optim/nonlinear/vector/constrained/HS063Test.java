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

public class HS063Test {
    private static final double pi = FastMath.PI;

    private static class HS063Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 3; }
        @Override public double value(RealVector x) {
            return (((((1000 - FastMath.pow(x.getEntry(0), 2)) - (2 * FastMath.pow(x.getEntry(1), 2))) - FastMath.pow(x.getEntry(2), 2)) - (x.getEntry(0) * x.getEntry(1))) - (x.getEntry(0) * x.getEntry(2)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS063Eq extends EqualityConstraint {
        HS063Eq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ ((((8 * x.getEntry(0)) + (14 * x.getEntry(1))) + (7 * x.getEntry(2)))) - (56), (((FastMath.pow(x.getEntry(0), 2) + FastMath.pow(x.getEntry(1), 2)) + FastMath.pow(x.getEntry(2), 2))) - (25) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 3; }
    }

    @Test
    public void testHS063() {
        InitialGuess guess = new InitialGuess(new double[]{2, 2, 2});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = 961.7151721;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS063Obj()), new HS063Eq());
        assertEquals(val, sol.getValue(), 1e-3);
    }
}