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

public class HS045Test {
    private static final double pi = FastMath.PI;

    private static class HS045Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 5; }
        @Override public double value(RealVector x) {
            return (2.0 - (((((x.getEntry(0) * x.getEntry(1)) * x.getEntry(2)) * x.getEntry(3)) * x.getEntry(4)) / 120.0));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS045Ineq extends InequalityConstraint {
        HS045Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (x.getEntry(0)) - (0.0), (1.0) - (x.getEntry(0)), (x.getEntry(1)) - (0.0), (2.0) - (x.getEntry(1)), (x.getEntry(2)) - (0.0), (3.0) - (x.getEntry(2)), (x.getEntry(3)) - (0.0), (4.0) - (x.getEntry(3)), (x.getEntry(4)) - (0.0), (5.0) - (x.getEntry(4)) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 5; }
    }

    @Test
    public void testHS045() {
        InitialGuess guess = new InitialGuess(new double[]{0.5, 1.0, 1.5, 2.0, 2.5});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = 1.0;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS045Obj()), new HS045Ineq());
        assertEquals(val, sol.getValue(), 1e-3);
    }
}