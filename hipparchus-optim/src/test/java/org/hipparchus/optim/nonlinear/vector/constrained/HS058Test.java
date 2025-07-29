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

public class HS058Test {
    private static final double pi = FastMath.PI;

    private static class HS058Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 2; }
        @Override public double value(RealVector x) {
            return ((100.0 * FastMath.pow((x.getEntry(1) - FastMath.pow(x.getEntry(0), 2)), 2)) + FastMath.pow((1 - x.getEntry(0)), 2));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    

    private static class HS058Ineq extends InequalityConstraint {
        HS058Ineq() { super(new ArrayRealVector(new double[]{  0.0, 0.0,0.0,0.0,0.0  })); }
        @Override public RealVector value(RealVector x) {
            return new ArrayRealVector(new double[]{
            FastMath.pow(x.getEntry(1), 2) - x.getEntry(0),                      // x2^2 - x1 ≥ 0
            FastMath.pow(x.getEntry(0), 2) - x.getEntry(1),                      // x1^2 - x2 ≥ 0
            FastMath.pow(x.getEntry(0), 2) + FastMath.pow(x.getEntry(1), 2) - 1, // x1^2 + x2^2 - 1 ≥ 0
            x.getEntry(0) - (-2.0),                                              // x1 ≥ -2
            0.5 - x.getEntry(0)                                                  // x1 ≤ 0.5
        });
        }
        @Override public RealMatrix jacobian(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public int dim() { return 2; }
    }

    @Test
    public void testHS058() {
         SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-10);
        InitialGuess guess = new InitialGuess(new double[]{-2.0, 1.0});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        optimizer.setDebugPrinter(System.out::println);
        double val = 3.19033354957;
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS058Obj()), new HS058Ineq());
        System.out.println(sol.getX()+";"+sol.getLambda());
        assertEquals(val, sol.getValue(), 1e-1);
        
    }
}