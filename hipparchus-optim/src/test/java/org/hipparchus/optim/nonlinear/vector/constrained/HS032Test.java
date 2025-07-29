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

public class HS032Test {
    private static final double pi = FastMath.PI;

    private static class HS032Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 3; }
        @Override public double value(RealVector x) {
            return (FastMath.pow(((x.getEntry(0) + (3.0 * x.getEntry(1))) + x.getEntry(2)), 2) + (4.0 * FastMath.pow((x.getEntry(0) - x.getEntry(1)), 2)));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    private static class HS032Eq extends EqualityConstraint {
        HS032Eq() { super(new ArrayRealVector(new double[]{ 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ (((x.getEntry(0) + x.getEntry(1)) + x.getEntry(2))) - (1.0) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 3; }
    }

    private static class HS032Ineq extends InequalityConstraint {
        HS032Ineq() { super(new ArrayRealVector(new double[]{ 0.0, 0.0, 0.0, 0.0 })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{ ((((6.0 * x.getEntry(1)) + (4.0 * x.getEntry(2))) - FastMath.pow(x.getEntry(0), 3))) - (3.0), (x.getEntry(0)) - (0.0), (x.getEntry(1)) - (0.0), (x.getEntry(2)) - (0.0) });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 3; }
    }

    @Test
    public void testHS032() {
        InitialGuess guess = new InitialGuess(new double[]{0.1, 0.7, 0.2});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
         SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-7);
        optimizer.setDebugPrinter(System.out::println);
        double val = 1.0;
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS032Obj()), new HS032Eq(), new HS032Ineq());
        assertEquals(val, sol.getValue(), 1e-6);
    }
}