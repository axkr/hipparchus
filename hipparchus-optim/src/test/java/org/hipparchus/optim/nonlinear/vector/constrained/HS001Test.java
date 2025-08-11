package org.hipparchus.optim.nonlinear.vector.constrained;



import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.optim.nonlinear.vector.constrained.LagrangeSolution;
import org.hipparchus.optim.nonlinear.vector.constrained.SQPOption;
import org.hipparchus.optim.nonlinear.vector.constrained.TwiceDifferentiableFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;


public class HS001Test {
    private static final double pi = FastMath.PI;

    private static class HS001Obj extends TwiceDifferentiableFunction {
        @Override public int dim() { return 2; }
        @Override public double value(RealVector x) {
            return ((100 * FastMath.pow((x.getEntry(1) - FastMath.pow(x.getEntry(0), 2)), 2)) + FastMath.pow((1 - x.getEntry(0)), 2));
        }
        @Override public RealVector gradient(RealVector x) { throw new UnsupportedOperationException(); }
        @Override public RealMatrix hessian(RealVector x) { throw new UnsupportedOperationException(); }
    }

    @Test
    public void testHS001() {
         SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-7);
        InitialGuess guess = new InitialGuess(new double[]{-2, 1});
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        
        double val = 0.0;
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS001Obj()));
        
        assertEquals(val, sol.getValue(), 1e-6);
    }
}