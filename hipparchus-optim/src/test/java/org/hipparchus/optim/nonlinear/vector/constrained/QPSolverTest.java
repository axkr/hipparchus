package org.hipparchus.optim.nonlinear.vector.constrained;









import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.optim.nonlinear.vector.constrained.LagrangeSolution;
import org.hipparchus.optim.nonlinear.vector.constrained.LinearEqualityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.LinearInequalityConstraint;
import org.hipparchus.optim.nonlinear.vector.constrained.QuadraticFunction;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;


public class QPSolverTest {

    @Test
    public void testGoldfarbIdnaniExample() {
        
        QuadraticFunction q = new QuadraticFunction(new double[][] { { 4.0, -2.0 }, { -2.0, 4.0 } },
                                                    new double[] { 6.0, 0.0 },
                                                    0.0);

        // y = 1
        LinearEqualityConstraint eqc = new LinearEqualityConstraint(new double[][] { { 1.0, 1.0 } },
                                                                    new double[] { 3.0 });


        // x > 0, y > 0, x + y > 2
        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 } },
                                                                          new double[] { 0.0, 0.0, 2.0 });
       

       

       

        
        QPDualActiveSolver solver = new QPDualActiveSolver();

        LagrangeSolution solution = solver.optimize(new ObjectiveFunction(q),eqc,ineqc);

        RealVector x = solution.getX();

        double expectedObj =solution.getValue();

        assertArrayEquals(new double[]{1.0, 2.0}, x.toArray(), 1e-8);
        assertEquals(12.0, expectedObj, 1e-8);
    }
}
