package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.optim.nonlinear.vector.constrained.LagrangeSolution;
import org.hipparchus.optim.nonlinear.vector.constrained.EqualityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.InequalityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.SQPOption;
import org.hipparchus.optim.nonlinear.vector.constrained.TwiceDifferentiableFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

public class HS038Test {
    private static final double pi = FastMath.PI;

    private static class HS038Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 4; }
        @Override public double value(RealVector x) {
            return ((((((100.0 * FastMath.pow((x.getEntry(1) - FastMath.pow(x.getEntry(0), 2.0)), 2.0)) + FastMath.pow((1.0 - x.getEntry(0)), 2.0)) + (90.0 * FastMath.pow((x.getEntry(3) - FastMath.pow(x.getEntry(2), 2.0)), 2.0))) + FastMath.pow((1.0 - x.getEntry(2)), 2.0)) + (10.1 * (FastMath.pow((x.getEntry(1) - 1.0), 2.0) + FastMath.pow((x.getEntry(3) - 1.0), 2.0)))) + ((19.8 * (x.getEntry(1) - 1.0)) * (x.getEntry(3) - 1.0)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS038Ineq extends InequalityConstraint {
        HS038Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (x.getEntry(0)) - (-10.0), (10.0) - (x.getEntry(0)), (x.getEntry(1)) - (-10.0), (10.0) - (x.getEntry(1)), (x.getEntry(2)) - (-10.0), (10.0) - (x.getEntry(2)), (x.getEntry(3)) - (-10.0), (10.0) - (x.getEntry(3)) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 4; }
    }

    @Test
    public void testHS038() {
        SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-7);
        
        InitialGuess guess = new InitialGuess(new double[]{-10.0, 10.0, 10.0, -10.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        
        double val = 0.0;
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS038Obj()), new HS038Ineq());
        assertEquals(val, sol.getValue(), 1e-3);
    }
}