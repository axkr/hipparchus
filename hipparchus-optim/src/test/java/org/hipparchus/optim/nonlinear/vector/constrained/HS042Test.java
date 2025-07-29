

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

public class HS042Test {
    private static final double pi = FastMath.PI;

    private static class HS042Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 4; }
        @Override public double value(RealVector x) {
            return (((FastMath.pow((x.getEntry(0) - 1), 2) + FastMath.pow((x.getEntry(1) - 2), 2)) + FastMath.pow((x.getEntry(2) - 3), 2)) + FastMath.pow((x.getEntry(3) - 4), 2));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS042Eq extends EqualityConstraint {
        HS042Eq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (x.getEntry(0)) - (2), ((FastMath.pow(x.getEntry(2), 2) + FastMath.pow(x.getEntry(3), 2))) - (2) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 4; }
    }

    @Test
    public void testHS042() {
        InitialGuess guess = new InitialGuess(new double[]{1, 1, 1, 1});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = (28 - (10 * FastMath.sqrt(2)));
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS042Obj()), new HS042Eq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}