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

public class HS064Test {
    private static final double pi = FastMath.PI;

    private static class HS064Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 3; }
        @Override public double value(RealVector x) {
            return ((((((5.0 * x.getEntry(0)) + (50000.0 / x.getEntry(0))) + (20.0 * x.getEntry(1))) + (72000.0 / x.getEntry(1))) + (10.0 * x.getEntry(2))) + (144000.0 / x.getEntry(2)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS064Ineq extends InequalityConstraint {
        HS064Ineq() { super(new ArrayRealVector(new double[]{ 0.0,0.0,0.0,0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (1.0) - ((((4.0 / x.getEntry(0)) + (32.0 / x.getEntry(1))) + (120.0 / x.getEntry(2)))) ,
                                                         x.getEntry(0)-1.0e-5,x.getEntry(1)-1.0e-5,x.getEntry(2)-1.0e-5     });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 3; }
    }

    @Test
    public void testHS064() {
        SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-11);
        InitialGuess guess = new InitialGuess(new double[]{1.0 ,1.0, 1.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = 6299.842428;
        LagrangeSolution sol = optimizer.optimize(guess, new ObjectiveFunction(new HS064Obj()), new HS064Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}