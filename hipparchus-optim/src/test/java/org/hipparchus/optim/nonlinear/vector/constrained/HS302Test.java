package org.hipparchus.optim.nonlinear.vector.constrained;



import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.InitialGuess;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.util.FastMath;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;


public class HS302Test {
    private static final double pi = FastMath.PI;

    private static class HS302Obj extends TwiceDifferentiableFunction {

    @Override
    public int dim() {
        return 100;
    }

    @Override
    public double value(RealVector x) {
        int n = dim();
        double fx = FastMath.pow(x.getEntry(0), 2) - 2.0 * x.getEntry(0);
        for (int i = 1; i < n; i++) {
            double xi = x.getEntry(i);
            double xim1 = x.getEntry(i - 1);
            fx += 2.0 * xi * xi - 2.0 * xim1 * xi;
        }
        return fx;
    }

    @Override
    public RealVector gradient(RealVector x) {
        int n = dim();
        double[] grad = new double[n];

        // i = 0
        grad[0] = 2.0 * x.getEntry(0) - 2.0 * x.getEntry(1) - 2.0;

        // i = 1 to n - 2
        for (int i = 1; i < n - 1; i++) {
            grad[i] = 4.0 * x.getEntry(i) - 2.0 * x.getEntry(i - 1) - 2.0 * x.getEntry(i + 1);
        }

        // i = n - 1
        grad[n - 1] = 4.0 * x.getEntry(n - 1) - 2.0 * x.getEntry(n - 2);

        return new ArrayRealVector(grad, false);
    }

    @Override
    public RealMatrix hessian(RealVector x) {
        return null;
    }
}

    @Test
    public void testHS302() {
         SQPOption sqpOption=new SQPOption();
        sqpOption.setMaxLineSearchIteration(20);
        sqpOption.setB(0.5);
        sqpOption.setMu(1.0e-4);
        sqpOption.setEps(10e-11);
        InitialGuess guess = new InitialGuess(new double[100]);
        
        SQPOptimizerS2 optimizer = new SQPOptimizerS2();
        optimizer.setDebugPrinter(System.out::println);
        
        double val = -100.0;
        LagrangeSolution sol = optimizer.optimize(sqpOption,guess, new ObjectiveFunction(new HS302Obj()));
        
        assertEquals(val, sol.getValue(), 1e-6);
    }
}