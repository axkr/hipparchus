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

public class HS037Test {
    private static final double pi = FastMath.PI;

    private static class HS037Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 3; }
        @Override public double value(RealVector x) {
            return (((-x.getEntry(0)) * x.getEntry(1)) * x.getEntry(2));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS037Ineq extends InequalityConstraint {
        HS037Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (72.0) - (((x.getEntry(0) + (2.0 * x.getEntry(1))) + (2.0 * x.getEntry(2)))), (((x.getEntry(0) + (2.0 * x.getEntry(1))) + (2.0 * x.getEntry(2)))) - (0.0) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 3; }
    }

    @Test
    public void testHS037() {
        InitialGuess guess = new InitialGuess(new double[]{10.0, 10.0, 10.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = -3456.0;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS037Obj()), new HS037Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}