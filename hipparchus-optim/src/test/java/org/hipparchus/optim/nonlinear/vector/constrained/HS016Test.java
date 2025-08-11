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

public class HS016Test {
    private static final double pi = FastMath.PI;
   
    //from the book 
    private static class HS016Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 1; }
        @Override public double value(RealVector x) {
            return FastMath.pow(x.getEntry(0) - 0.25, 2.0);
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS016Ineq extends InequalityConstraint {
        HS016Ineq() { super(new ArrayRealVector(new double[]{ 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{  (x.getEntry(0))  });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 1; }
    }

    @Test
    public void testHS016() {
        SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-7);
        InitialGuess guess = new InitialGuess(new double[]{1.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        double val = (0.0);
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS016Obj()), new HS016Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}