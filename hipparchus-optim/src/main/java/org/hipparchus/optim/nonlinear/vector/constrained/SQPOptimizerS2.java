/*
 * Licensed to the Hipparchus project under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The Hipparchus project licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.hipparchus.optim.nonlinear.vector.constrained;

import org.hipparchus.linear.Array2DRowRealMatrix;
import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.MatrixUtils;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.Precision;

/**
 * Sequential Quadratic Programming Optimizer (extended version).
 *
 * <p>
 * Minimizes a nonlinear objective function subject to equality, inequality, and
 * box constraints using a Sequential Quadratic Programming method. This
 * implementation is inspired by the algorithm described in: "On the convergence
 * of a sequential quadratic programming method" by Klaus Schittkowski (1982).
 * </p>
 *
 * <p>
 * Supports: equality constraints, inequality constraints,
 * penalty function updates, line search strategies, BFGS Hessian update, and
 * augmented QP formulation using a relaxation variable.</p>
 *
 * @since 3.1
 */
public class SQPOptimizerS2 extends AbstractSQPOptimizer2 {

    
    SQPLogger formatter = new SQPLogger();

    /**
     * Value of the equality constraints.
     */
    private RealVector eqEval;

    /**
     * Value of the inequality constraints.
     */
    private RealVector ineqEval;

    /**
     * Gradient of the objective function.
     */
    private RealVector J;

    /**
     * Hessian approximation.
     */
    private RealMatrix H;

    /**
     * Jacobian of the inequality constraints.
     */
    private RealMatrix JI;

    /**
     * Jacobian of the equality constraints.
     */
    private RealMatrix JE;
    private double functionEval;
    private RealVector x;

    /**
     * {@inheritDoc}
     */
    @Override
    public LagrangeSolution doOptimize() {
        formatter.setEps(getSettings().getEps());
        formatter.logHeader();
        int me = 0;
        int mi = 0;

        //EQUALITY CONSTRAINT
        if (this.getEqConstraint() != null) {
            me = getEqConstraint().dimY();
        }
        //INEQUALITY CONSTRAINT
        if (this.getIqConstraint() != null) {
            mi = getIqConstraint().dimY();
        }
        final int m = me + mi;

        double alpha;
        double rho = 100.0;

       
        if (this.getStartPoint() != null) {
            x = new ArrayRealVector(this.getStartPoint());
        } else {
            x = new ArrayRealVector(this.getObj().dim());
        }

        RealVector y = new ArrayRealVector(me + mi, 0.0);

       //all the function and constraint evaluation will be performed inside the penalty function
        MeritFunctionL2 penalty = new MeritFunctionL2(this.getObj(), this.getEqConstraint(), this.getIqConstraint(), x);

        LineSearch lineSearch = new LineSearch(getSettings().getEps(), 5, getSettings().getMu(), getSettings().getB(),
                                               getSettings().getMaxLineSearchIteration(), 2);
        H = MatrixUtils.createRealIdentityMatrix(x.getDimension());
        
        BFGSUpdater bfgs = new BFGSUpdater(H, 1.0e-11, getMatrixDecompositionTolerance().getEpsMatrixDecomposition());
        //INITIAL VALUES

        functionEval = penalty.getObjEval();
        if (this.getEqConstraint() != null) {
            eqEval = penalty.getEqEval();
        }
        if (this.getIqConstraint() != null) {
            ineqEval = penalty.getIqEval();
        }
        
        switch(getSettings().getGradientMode()){
                 case 0:externalGradient(x);break;
                 case 1:forwardGradient(x);break;
                 case 2:centralGradient(x);break;
        }

        RealVector dx = new ArrayRealVector(x.getDimension());
        RealVector u = new ArrayRealVector(y.getDimension());
        penalty.update(J, JE, JI, x, y, dx, u);
        boolean augmented = true;

        
        boolean crit0 = false; 
        boolean crit1 = false; 
        boolean crit2 = false; 
        boolean crit3 = false; 
        for (int i = 0; i < this.getMaxIterations(); i++) {
            iterations.increment();

            
            LagrangeSolution sol1 = null;

            int qpLoop = 0;
            double sigma = getSettings().getSigmaMax() * 10.0;

            //log("SQP:hessian" + H);
            //LOOP TO FIND SOLUTION WITH SIGMA<SIGMA THRESHOLD

            while ((sigma > getSettings().getSigmaMax() || sigma < - Precision.EPSILON) && qpLoop < getSettings().getQpMaxLoop()) {
                sol1=augmented?solveAugmentedQP(x, y, rho):solveQP(x);
                sigma=(sol1.getX().getDimension()==0)? getSettings().getSigmaMax() * 10.0: sol1.getValue();
               
                
                if (sigma > getSettings().getSigmaMax() || sigma < - Precision.EPSILON) {
                    rho = getSettings().getRhoCons() * rho;
                    qpLoop += 1;
                    augmented = true;

                }

            }
            //IF SIGMA>SIGMA THRESHOLD ASSIGN DIRECTION FROM PENALTY GRADIENT

            if (qpLoop == getSettings().getQpMaxLoop()) {

                dx = (MatrixUtils.inverse(H).operate(penalty.gradX())).mapMultiply(-1.0);
                u = y.subtract(penalty.gradY());
                sigma = 0;
                augmented = true;

            } else {
                dx = sol1.getX();
                u = sol1.getLambda();
                sigma = sol1.getValue();
                penalty.updateRj(H,y, dx, u,sigma, iterations.getCount());
                //switch to normal QP if additional variabile is small enough
                if (FastMath.abs(sigma) < getSettings().getEps())  augmented = false;
               
            }

             penalty.update(J, JE, JI, x, y, dx, u);

            //if penalty gradinet is >=0 skip line search and rty again with augmented QP
            if (penalty.getGradient() < 0) {
               
                rho = updateRho(dx, u, H,JE,JI, sigma);


                //LINE SEARCH
                alpha = lineSearch.search(penalty);

                //STORE OLD DERIVATIVE VALUE
                RealVector JOLD = J;
                RealMatrix JIOLD = JI;
                RealMatrix JEOLD = JE;

                //OLD LAGRANGIANE GRADIENT UPDATE WITH NEW MULTIPLIER
                if (m > 0) {
                    y = y.add(u.subtract(y).mapMultiply(alpha));
                }
                RealVector lagOld = lagrangianGradX(JOLD, JEOLD, JIOLD, x, u);
                //UPDATE ALL VARIABLE FOR THE NEXT STEP
                x = x.add(dx.mapMultiply(alpha));
                //PENALTY STORE ALL VARIABLE CALCULATED WITH THE LAST STEP
                //penalty function memorize last calculation done in the line search
                functionEval = penalty.getObjEval();
                eqEval = penalty.getEqEval();
                ineqEval = penalty.getIqEval();
                //calculate new Gradients
                switch(getSettings().getGradientMode()){
                 case 0:externalGradient(x);break;
                 case 1:forwardGradient(x);break;
                 case 2:centralGradient(x);break;    
                }
                //internalGradientCentered(x);
                //NEW LAGRANGIANE GRADIENT UPDATE WITH NEW MULTIPLIER AND NEW VALUES
                RealVector lagnew = lagrangianGradX(J, JE, JI, x, u);

//            //CONVERGENCE CHECK 
                //STATIONARIETY
                double kkt=lagnew.getNorm();
                double step1=dx.mapMultiply(alpha).dotProduct(H.operate(dx.mapMultiply(alpha)));
                double step2= alpha * dx.getNorm();
                double violation=constraintViolation();
                crit0 = kkt<=FastMath.sqrt(getSettings().getEps());
                crit1 = step1<=getSettings().getEps() * getSettings().getEps();
                crit2 = step2<= getSettings().getEps()*(1.0 + x.getNorm());
                crit3= violation<=FastMath.sqrt(getSettings().getEps());
                formatter.logRow(iterations.getCount(), alpha,lineSearch.getIteration(),step2, step1,kkt, violation, sigma, penalty.getPenaltyEval(), functionEval);
                
                if ((crit0 || (crit1 && crit2)) && crit3) {
                   
                   //System.out.println("SQP:CONVERGENCE REACHED:"+"STATIONARIETY:"+crit0+";STEP SIZE:"+crit1+"and"+crit2+";VIOLATION:"+crit3);
                   break;
                }

                //HESSIAN UPDATE WITH THE LOGIC OF LINE SEARCH AND WITH THE INTERNAL LOGIC(DUMPING)
                if (lineSearch.isBadStepFailed()) {
                    //reset hessain and inizialize for augmented QP solution with multiplier to zero
                    // bfgs.resetHessian(FastMath.min(gamma,10e6));
                    bfgs.resetHessian();
                    
                    H = bfgs.getHessian();
                    augmented = true;
                    rho = 100.0;
                    penalty.resetRj();
                    //y.set(0.0);
                    lineSearch.resetBadStepCount();

                } else if (lineSearch.isBadStepDetected()) {
                    //System.out.println("SQP:BAD STEP"+lineSearch.getBadStepCount());
                   
                    //mantain the same Hessian
                    
                    H = bfgs.getHessian();

                } else {
                    // good step detected procede with hessian update
                    bfgs.update(dx.mapMultiply(alpha), lagnew.subtract(lagOld));
                    H = bfgs.getHessian();

                }
               
               
                
                
//                System.out.println("   SQP ITERATION:LAMDA:" + y);

            } else {
                augmented = true;
                rho = getSettings().getRhoCons() * rho;
            }
        }
       
         formatter.logRow(-1,null,null,crit2,crit1,crit0,crit3,null, null,null);
     
        
        return new LagrangeSolution(x, y, functionEval);
    }

    private double constraintViolation() {

        double crit = 0;
        if (this.getEqConstraint() != null) {
            crit = crit + eqEval.subtract(this.getEqConstraint().getLowerBound()).getL1Norm();
        }
        if (this.getIqConstraint() != null) {
            RealVector violated = ineqEval.subtract(this.getIqConstraint().getLowerBound());
            for (int k = 0; k < violated.getDimension(); k++) {
                violated.setEntry(k, FastMath.min(0.0, violated.getEntry(k)));
            }
            crit = crit + violated.getL1Norm();
        }
        return crit;
    }

   

    private double updateRho(RealVector dx, RealVector dy, RealMatrix H, RealMatrix JE,RealMatrix JI, double additionalVariable) {
        int me = JE != null ? JE.getRowDimension() : 0;
        int mi = JI != null ? JI.getRowDimension() : 0;
        RealMatrix JAC;
        if (me + mi > 0) {
            JAC = new Array2DRowRealMatrix(me + mi, x.getDimension());
                if (JE != null) {
                    JAC.setSubMatrix(JE.getData(), 0, 0);
                }
                if (JI != null) {
                    JAC.setSubMatrix(JI.getData(), me, 0);
                }

            double num = 10.0 * FastMath.pow(dx.dotProduct(JAC.preMultiply(dy)), 2);
            double den = (1.0 - additionalVariable) * (1.0 - additionalVariable) * dx.dotProduct(H.operate(dx));
            //double den = (1.0 - additionalVariable) * dx.dotProduct(H.operate(dx));

            return FastMath.max(10.0, num / den);

        }
        return 0;
    }
/**
 * Solves an augmented QP subproblem by introducing a relaxation variable σ (sigma)
 * when equality constraints exist or inequality constraints are violated.
 * <p>
 * The relaxation allows for infeasibility in the constraints, controlled by a penalty {@code rho}.
 * The augmented problem is of the form:
 * <pre>
 *   minimize   ½·dxᵗ·H·dx + gᵗ·dx + ½·rho·σ²
 *   subject to:
 *       JE·dx - σ = ce(x) - bₑ   (equality constraints)
 *       JI·dx - σ ≥ ci(x) - bᵢ   (inequality constraints if violated)
 *       σ ≥ 0
 * </pre>
 * The slack σ is activated only if:
 * <ul>
 *   <li>equality constraints are present, or</li>
 *   <li>inequality constraints are active/violated or their multipliers are positive</li>
 * </ul>
 *
 * @param x   current point in primal space
 * @param y   current dual vector (used to detect activity of constraints)
 * @param rho penalty parameter for the relaxation variable
 * @return QP solution with optional relaxation variable σ; {@code null} if solver fails
 */
//private LagrangeSolution solveQP(RealVector x, RealVector y, double rho) {
//
//    RealMatrix H = this.H;
//    RealVector g = this.J;
//    int n = x.getDimension();
//    int me = getEqConstraint() != null ? getEqConstraint().dimY() : 0;
//    int mi = getIqConstraint() != null ? getIqConstraint().dimY() : 0;
//
//    // Determine if sigma relaxation is needed
//    boolean violated = false;
//    if (mi > 0) {
//        RealVector violation = ineqEval.subtract(getIqConstraint().getLowerBound());
//        violated = violation.getMinValue() <= getSettings().getEps() || y.getMaxValue() > 0;
//    }
//    int add = (me > 0 || violated) ? 1 : 0;
//
//    // Augmented Hessian
//    RealMatrix H1 = new Array2DRowRealMatrix(n + add, n + add);
//    H1.setSubMatrix(H.getData(), 0, 0);
//    if (add == 1) {
//        H1.setEntry(n, n, rho); // Penalize sigma
//    }
//
//    // Augmented gradient
//    RealVector g1 = new ArrayRealVector(n + add);
//    g1.setSubVector(0, g); // sigma component is zero by default
//
//    // Equality constraints (augmented)
//    LinearEqualityConstraint eqc = null;
//    if (me > 0) {
//        RealMatrix eqJacob = this.JE;
//        RealMatrix Ae = new Array2DRowRealMatrix(me, n + add);
//        RealVector be = getEqConstraint().getLowerBound().subtract(eqEval);
//
//        Ae.setSubMatrix(eqJacob.getData(), 0, 0);
//        RealVector deltaEq = eqEval.subtract(getEqConstraint().getLowerBound());
//        Ae.setColumnVector(n, deltaEq.mapMultiply(-1.0));
//
//        eqc = new LinearEqualityConstraint(Ae, be);
//    }
//
//    // Inequality constraints (augmented if needed)
//    LinearInequalityConstraint iqc = null;
//    if (mi > 0) {
//        RealMatrix iqJacob = this.JI;
//        RealMatrix Ai = new Array2DRowRealMatrix(mi, n + add);
//        RealVector bi = getIqConstraint().getLowerBound().subtract(ineqEval);
//        Ai.setSubMatrix(iqJacob.getData(), 0, 0);
//
//        if (add == 1) {
//            RealVector conditioniq = ineqEval.subtract(getIqConstraint().getLowerBound());
//            for (int i = 0; i < conditioniq.getDimension(); i++) {
//                if (!(conditioniq.getEntry(i) <= getSettings().getEps() || y.getEntry(me + i) > 0)) {
//                    conditioniq.setEntry(i, 0.0);
//                }
//            }
//            Ai.setColumnVector(n, conditioniq.mapMultiply(-1.0));
//        }
//
//        iqc = new LinearInequalityConstraint(Ai, bi);
//    }
//
//    // Box constraints (relaxed only if sigma is present)
//    LinearBoundedConstraint bc = null;
//    if (add == 1) {
//        
//        RealMatrix sigmaA = new Array2DRowRealMatrix(1 ,n+1);
//        sigmaA.setEntry(0, n, 1.0); // σ ≥ 0
//
//        ArrayRealVector lb = new ArrayRealVector(1 , 0.0);
//        ArrayRealVector ub = new ArrayRealVector(1 , 1.0);
//
//        // (Optional) Add box constraints below
//        // if (box > 0) {
//        //     sigmaA.setSubMatrix(getBoxConstraint().jacobian(x).getData(), 1, 0);
//        //     lb.setSubVector(1, getBoxConstraint().getLowerBound().subtract(x));
//        //     ub.setSubVector(1, getBoxConstraint().getUpperBound().subtract(x));
//        // }
//
//        bc = new LinearBoundedConstraint(sigmaA, lb, ub);
//    }
//
//    // QP solve
//    QuadraticFunction q = new QuadraticFunction(H1, g1, 0.0);
//    LagrangeSolution sol = getQPSolver().optimize(new ObjectiveFunction(q), iqc, eqc, bc);
//    if (sol == null) {
//        return null;
//    }
//
//    double sigma = (add == 1) ? sol.getX().getEntry(n) : 0.0;
//    RealVector solutionX = sol.getX().getSubVector(0, n);
//    RealVector lambda = (me + mi > 0) ? sol.getLambda().getSubVector(0, me + mi) : new ArrayRealVector(0,0);
//
//    return new LagrangeSolution(solutionX, lambda, sigma);
//}

    private LagrangeSolution solveAugmentedQP(RealVector x, RealVector y, double rho) {

        RealMatrix H = this.H;
        RealVector g = J;

        int me = 0;
        int mi = 0;
        int add = 0;
        boolean violated = false;
        if (getEqConstraint() != null) {
            me = getEqConstraint().dimY();
        }
        if (getIqConstraint() != null) {

            mi = getIqConstraint().dimY();
            violated = ineqEval.subtract(getIqConstraint().getLowerBound()).getMinValue() <= getSettings().getEps()
                    || y.getMaxValue() > 0;

        }
        // violated = true;
        if (me > 0 || violated) {
            add = 1;

        }

        RealMatrix H1 = new Array2DRowRealMatrix(H.getRowDimension() + add, H.getRowDimension() + add);
        H1.setSubMatrix(H.getData(), 0, 0);
        if (add == 1) {
            H1.setEntry(H.getRowDimension(), H.getRowDimension(), rho);
        }

        RealVector g1 = new ArrayRealVector(g.getDimension() + add);
        g1.setSubVector(0, g);

        LinearEqualityConstraint eqc = null;
        RealVector conditioneq;
        if (getEqConstraint() != null) {
            //RealMatrix eqJacob = constraintJacob.getSubMatrix(0, me - 1, 0, x.getDimension() - 1);
            RealMatrix eqJacob = JE;
            RealMatrix Ae = new Array2DRowRealMatrix(me, x.getDimension() + add);
            RealVector be = new ArrayRealVector(me);
            Ae.setSubMatrix(eqJacob.getData(), 0, 0);
            conditioneq = this.eqEval.subtract(getEqConstraint().getLowerBound());
            Ae.setColumnVector(x.getDimension(), conditioneq.mapMultiply(-1.0));

            be.setSubVector(0, getEqConstraint().getLowerBound().subtract(this.eqEval));
            eqc = new LinearEqualityConstraint(Ae, be);

        }
        LinearInequalityConstraint iqc = null;

        if (getIqConstraint() != null) {
//
            // RealMatrix iqJacob = constraintJacob.getSubMatrix(me, me + mi - 1, 0, x.getDimension() - 1);
            RealMatrix iqJacob = JI;
            RealMatrix Ai = new Array2DRowRealMatrix(mi, x.getDimension() + add);
            RealVector bi = new ArrayRealVector(mi);
            Ai.setSubMatrix(iqJacob.getData(), 0, 0);

            RealVector conditioniq = this.ineqEval.subtract(getIqConstraint().getLowerBound());

            if (add == 1) {

                for (int i = 0; i < conditioniq.getDimension(); i++) {

                    if (!(conditioniq.getEntry(i) <= getSettings().getEps() || y.getEntry(me + i) > 0)) {
                        conditioniq.setEntry(i, 0);
                    }
                }

                Ai.setColumnVector(x.getDimension(), conditioniq.mapMultiply(-1.0));

            }
            bi.setSubVector(0, getIqConstraint().getLowerBound().subtract(this.ineqEval));
            iqc = new LinearInequalityConstraint(Ai, bi);

        }
        int box = 0;
        if (getBoxConstraint() != null) {
            box = getBoxConstraint().dimY();
        }
        //this.log("MI:" + box);
        LinearBoundedConstraint bc = null;

        if (add == 1) {

            RealMatrix sigmaA = new Array2DRowRealMatrix(1 + box, x.getDimension() + 1);
            sigmaA.setEntry(0, x.getDimension(), 1.0);

            ArrayRealVector lb = new ArrayRealVector(1 + box, 0.0);
            ArrayRealVector ub = new ArrayRealVector(1 + box, 1.0);
            bc = new LinearBoundedConstraint(sigmaA, lb, ub);

        }

        QuadraticFunction q = new QuadraticFunction(H1, g1, 0);
       
        LagrangeSolution sol = this.getQPSolver().optimize(new ObjectiveFunction(q), iqc, eqc, bc);

        // Solve the QP problem
   
    if (sol.getX().getDimension() == 0) {
        return sol;
    }
        double sigma;
        if (add == 1) {
            sigma = sol.getX().getEntry(x.getDimension());
        } else {
            sigma = 0;
        }
        //System.out.println("QP MULTIPLIER:" + sol.getLambda());
        return (me + mi == 0) ?
               new LagrangeSolution(sol.getX().getSubVector(0, x.getDimension()), null, sigma) :
               new LagrangeSolution(sol.getX().getSubVector(0, x.getDimension()), sol.getLambda().getSubVector(0, me + mi), sigma);
    }

    /**
     * Solves the Quadratic Programming (QP) subproblem in the current SQP iteration.
     * @param x the current point in the primal space
     * @return a {@link LagrangeSolution} representing the QP solution, or {@code null} if the QP failed
     */
private LagrangeSolution solveQP(RealVector x) {

    QuadraticFunction q = new QuadraticFunction(this.H, this.J, 0);
    int n = x.getDimension();
    int me = 0;
    int mi = 0;

    // Equality constraints
    LinearEqualityConstraint eqc = null;
    if (getEqConstraint() != null) {
        me = getEqConstraint().dimY();
        RealMatrix Ae = new Array2DRowRealMatrix(me, n);
        RealVector be = getEqConstraint().getLowerBound().subtract(eqEval);
        Ae.setSubMatrix(JE.getData(), 0, 0);
        eqc = new LinearEqualityConstraint(Ae, be);
    }

    // Inequality constraints
    LinearInequalityConstraint iqc = null;
    if (getIqConstraint() != null) {
        mi = getIqConstraint().dimY();
        RealMatrix Ai = new Array2DRowRealMatrix(mi, n);
        RealVector bi = getIqConstraint().getLowerBound().subtract(ineqEval);
        Ai.setSubMatrix(JI.getData(), 0, 0);
        iqc = new LinearInequalityConstraint(Ai, bi);
    }

    // Solve the QP problem
    LagrangeSolution sol = getQPSolver().optimize(new ObjectiveFunction(q), iqc, eqc);
    if (sol.getX().getDimension() == 0) {
        return sol;
    }

    // Extract primal and dual components
    RealVector solutionX = sol.getX().getSubVector(0, n);
    RealVector solutionLambda = (me + mi > 0) ? sol.getLambda().getSubVector(0, me + mi) : new ArrayRealVector(0,0);

    return new LagrangeSolution(solutionX, solutionLambda, 0.0);
}


  

    
 /**
 * Computes the gradient of the Lagrangian function with respect to the primal variable {@code x}.
 * <p>
 * The Lagrangian is defined as:
 * <pre>
 *     L(x, y) = f(x) - yₑᵗ·cₑ(x) - yᵢᵗ·cᵢ(x)
 * </pre>
 * where:
 * <ul>
 *   <li>{@code f(x)} is the objective function</li>
 *   <li>{@code cₑ(x)} are the equality constraints</li>
 *   <li>{@code cᵢ(x)} are the inequality constraints</li>
 *   <li>{@code y = [yₑ; yᵢ]} is the vector of Lagrange multipliers</li>
 * </ul>
 * The gradient with respect to {@code x} is given by:
 * <pre>
 *     ∇ₓ L(x, y) = ∇f(x) - JEᵗ·yₑ - JIᵗ·yᵢ
 * </pre>
 *
 * @param J  the gradient of the objective function {@code ∇f(x)}, length {@code n}
 * @param JE the Jacobian of the equality constraints, shape {@code [me x n]} (nullable)
 * @param JI the Jacobian of the inequality constraints, shape {@code [mi x n]} (nullable)
 * @param x  the current point in the primal space (not used directly, included for API symmetry)
 * @param y  the stacked Lagrange multipliers {@code [yₑ; yᵢ]}, length {@code me + mi}
 * @return the gradient of the Lagrangian with respect to {@code x}, length {@code n}
 */
public RealVector lagrangianGradX(final RealVector J,
                                  final RealMatrix JE,
                                  final RealMatrix JI,
                                  final RealVector x,
                                  final RealVector y) {

    RealVector gradL =new ArrayRealVector(J);
    int offset = 0;

    // Subtract JEᵗ · yₑ if equality constraints exist
    if (JE != null) {
        int me = JE.getRowDimension();
        RealVector yEq = y.getSubVector(0, me);
        RealVector termEq = JE.preMultiply(yEq);
        gradL=gradL.subtract(termEq);
        offset += me;
    }

    // Subtract JIᵗ · yᵢ if inequality constraints exist
    if (JI != null) {
        int mi = JI.getRowDimension();
        RealVector yIq = y.getSubVector(offset, mi);
        RealVector termIq = JI.preMultiply(yIq);
        gradL=gradL.subtract(termIq);
    }

    return gradL;
}

    
    
    private void externalGradient(RealVector x)
    {
        J = this.getObj().gradient(x);
        if (this.getEqConstraint() != null) {
            JE = this.getEqConstraint().jacobian(x);
        }
        if (this.getIqConstraint() != null) {
            JI = this.getIqConstraint().jacobian(x);
        }
        
    }

/**
 * Computes the gradient of the objective function and the Jacobians of the constraints
 * using forward finite differences (first-order accurate).
 * <p>
 * Each variable is perturbed independently by a small step size proportional to
 * the square root of machine precision, and partial derivatives are approximated
 * using forward differencing.
 * </p>
 *
 * @param point the current point at which to evaluate gradients (must correspond to
 *              the values stored in {@code functionEval}, {@code eqEval}, and {@code ineqEval})
 */
private void forwardGradient(RealVector point) {

    int n = point.getDimension();
    double sqrtEps = FastMath.sqrt(Precision.EPSILON);

    double fRef = this.functionEval;
    RealVector eqRef = this.eqEval;
    RealVector iqRef = this.ineqEval;

    RealVector gradF = new ArrayRealVector(n);
    RealMatrix gradEq = (getEqConstraint() != null) ? new Array2DRowRealMatrix(eqRef.getDimension(), n) : null;
    RealMatrix gradIq = (getIqConstraint() != null) ? new Array2DRowRealMatrix(iqRef.getDimension(), n) : null;

    for (int i = 0; i < n; i++) {
        double xi = point.getEntry(i);
        double h = sqrtEps * FastMath.max(1.0, FastMath.abs(xi));

        RealVector xPerturbed =new ArrayRealVector(point);
        xPerturbed.setEntry(i, xi + h);

        double fPerturbed = getObj().value(xPerturbed);
        gradF.setEntry(i, (fPerturbed - fRef) / h);

        if (gradEq != null) {
            RealVector eqPerturbed = getEqConstraint().value(xPerturbed);
            RealVector diffEq = eqPerturbed.subtract(eqRef).mapMultiply(1.0 / h);
            gradEq.setColumnVector(i, diffEq);
        }

        if (gradIq != null) {
            RealVector iqPerturbed = getIqConstraint().value(xPerturbed);
            RealVector diffIq = iqPerturbed.subtract(iqRef).mapMultiply(1.0 / h);
            gradIq.setColumnVector(i, diffIq);
        }
    }
 
        this.J=gradF;
       this.JE=gradEq;
       this.JI=gradIq;
}


 /**
 * Computes the gradient of the objective function and the Jacobians of the constraints
 * using centered finite differences (second-order accurate).
 * 
 * @param x the current point at which to evaluate gradients (must correspond to
 *          the values stored in {@code functionEval}, {@code eqEval}, and {@code ineqEval})
 */
private void centralGradient(RealVector x) {

    int n = x.getDimension();
    double hBase = FastMath.cbrt(Precision.EPSILON);

    double fPlus, fMinus;
    RealVector gradF = new ArrayRealVector(n);
    RealMatrix gradEq = (getEqConstraint() != null) ? new Array2DRowRealMatrix(eqEval.getDimension(), n) : null;
    RealMatrix gradIq = (getIqConstraint() != null) ? new Array2DRowRealMatrix(ineqEval.getDimension(), n) : null;

    for (int i = 0; i < n; i++) {
        double xi = x.getEntry(i);
        double h = hBase * FastMath.max(1.0, FastMath.abs(xi));

        RealVector xPlus = new ArrayRealVector(x);
        RealVector xMinus = new ArrayRealVector(x);
        xPlus.addToEntry(i, h);
        xMinus.addToEntry(i, -h);

        fPlus = getObj().value(xPlus);
        fMinus = getObj().value(xMinus);
        gradF.setEntry(i, (fPlus - fMinus) / (2.0 * h));

        if (gradEq != null) {
            RealVector eqPlus = getEqConstraint().value(xPlus);
            RealVector eqMinus = getEqConstraint().value(xMinus);
            RealVector dEq = eqPlus.subtract(eqMinus).mapDivide(2.0 * h);
            gradEq.setColumnVector(i, dEq);
        }

        if (gradIq != null) {
            RealVector iqPlus = getIqConstraint().value(xPlus);
            RealVector iqMinus = getIqConstraint().value(xMinus);
            RealVector dIq = iqPlus.subtract(iqMinus).mapDivide(2.0 * h);
            gradIq.setColumnVector(i, dIq);
        }
    }

    this.J = gradF;
    this.JE = gradEq;
    this.JI = gradIq;
}

  public void setDebugPrinter(DebugPrinter printer) {
    formatter.setDebugPrinter(printer);
  }
}
