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

public class HS044Test {
    private static final double pi = FastMath.PI;

    private static class HS044Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 4; }
        @Override public double value(RealVector x) {
            return ((((((x.getEntry(0) - x.getEntry(1)) - x.getEntry(2)) - (x.getEntry(0) * x.getEntry(2))) + (x.getEntry(0) * x.getEntry(3))) + (x.getEntry(1) * x.getEntry(2))) - (x.getEntry(1) * x.getEntry(3)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS044Ineq extends InequalityConstraint {
        HS044Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0,0,0,0})); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (8) - ((x.getEntry(0) + (2 * x.getEntry(1)))), (12) - (((4 * x.getEntry(0)) + x.getEntry(1))), (12) - (((3 * x.getEntry(0)) + (4 * x.getEntry(1)))), (8) - (((2 * x.getEntry(2)) + x.getEntry(3))), (8) - ((x.getEntry(2) + (2 * x.getEntry(3)))), (5) - ((x.getEntry(2) + x.getEntry(3))) ,
            x.getEntry(0),x.getEntry(1),x.getEntry(2),x.getEntry(3)});
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 4; }
    }

    @Test
    public void testHS044() {
        InitialGuess guess = new InitialGuess(new double[]{0, 0, 0, 0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = -15.0;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS044Obj()), new HS044Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}