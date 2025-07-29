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


public class HS009Test {
    private static final double pi = FastMath.PI;

    private static class HS009Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 2; }
        @Override public double value(RealVector x) {
            return (FastMath.sin(((pi * x.getEntry(0)) / 12)) * FastMath.cos(((pi * x.getEntry(1)) / 16)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS009Eq extends EqualityConstraint {
        HS009Eq() { super(new ArrayRealVector(new double[]{ 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (((4 * x.getEntry(0)) - (3 * x.getEntry(1)))) - (0) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 2; }
    }

    @Test
    public void testHS009() {
        InitialGuess guess = new InitialGuess(new double[]{0, 0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = -0.5;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS009Obj()), new HS009Eq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}