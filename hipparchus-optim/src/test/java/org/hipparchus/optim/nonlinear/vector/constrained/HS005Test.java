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


public class HS005Test {
    private static final double pi = FastMath.PI;

    private static class HS005Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 2; }
        @Override public double value(RealVector x) {
            return ((((FastMath.sin((x.getEntry(0) + x.getEntry(1))) + FastMath.pow((x.getEntry(0) - x.getEntry(1)), 2)) - (1.5 * x.getEntry(0))) + (2.5 * x.getEntry(1))) + 1);
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS005Ineq extends InequalityConstraint {
        HS005Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (x.getEntry(0)) - (-1.5), (4) - (x.getEntry(0)), (x.getEntry(1)) - (-3), (3) - (x.getEntry(1)) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 2; }
    }

    @Test
    public void testHS005() {
        InitialGuess guess = new InitialGuess(new double[]{0, 0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = -1.91322207;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS005Obj()), new HS005Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}