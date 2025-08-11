package org.hipparchus.optim.nonlinear.vector.constrained;



import org.hipparchus.linear.Array2DRowRealMatrix;
import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.nonlinear.vector.constrained.Constraint;
import org.hipparchus.optim.nonlinear.vector.constrained.TwiceDifferentiableFunction;
import org.hipparchus.util.FastMath;


/**
 * Augmented Penalty Function 
 *
 * This class computes the penalty function and its gradient, combining:
 * - The objective function
 * - Equality constraints
 * - Inequality constraints
 *
 * Typical usage:
 *  - Call update(...) before line search or Hessian update
 *  - Use value(alpha) to evaluate penalty at x + alpha * dx(this store also evalutation of objective and constraints)
 *  - Use gradient() to retrieve penalty gradient at current point
 */
public class MeritFunctionL2 {

    /** Objective function. */
    private final TwiceDifferentiableFunction objective;

    /** Equality constraints (may be null). */
    private final Constraint eqConstraint;

    /** Inequality constraints (may be null). */
    private final Constraint iqConstraint;

    // Cached parameters
//    private double currentF;
    private RealVector x;
    private RealVector y;
    private RealVector dx;
    private RealVector u;
    private RealVector r;
    private RealVector J;
    private double penaltyGradient;
    private double penaltyStart;
    private double objEval;
    private RealVector eqEval;
    private RealVector iqEval;
    private double pEval;
    private RealMatrix JE;
    private RealMatrix JI;
    private int evalCounter=0;
//    private RealVector eqVal;
//    private RealVector iqVal;
    private int m;
    
    /**
     * Constructor.
     *
     * @param objective Objective function
     * @param eqConstraint Equality constraint (may be null)
     * @param iqConstraint Inequality constraint (may be null)
     * @param x
     */
    public MeritFunctionL2(final TwiceDifferentiableFunction objective,
                           final Constraint eqConstraint,
                           final Constraint iqConstraint,RealVector x) {
        this.objective = objective;
        this.eqConstraint = eqConstraint;
        this.iqConstraint = iqConstraint;
        this.x=new ArrayRealVector(x);
        int me=0;
        int mi=0;
        if (this.eqConstraint!=null)me=this.eqConstraint.dimY();
        if (this.iqConstraint!=null)mi=this.iqConstraint.dimY();
        m=me+mi;
        this.dx=new ArrayRealVector(x.getDimension());
        this.y=new ArrayRealVector(m);
        this.u=new ArrayRealVector(m);
        this.r=new ArrayRealVector(m,1.0);
        this.J=new ArrayRealVector(x.getDimension());
//        this.JE=new Array2DRowRealMatrix(me,x.getDimension());
//        this.JI=new Array2DRowRealMatrix(mi,x.getDimension());
        this.eqEval=new ArrayRealVector(me);
        this.iqEval=new ArrayRealVector(mi);
        //this evaluate objective function contraints function and penaly
        penaltyStart=this.value(0);
    }

    /**
     * Update internal parameters for next penalty computation.
     *
     * 
     * @param J Gradient of objective at current x
     * @param JE Gradient of equality x
     * @param JI Gradient of inequality at current x
     * @param x Current iterate
     * @param y Lagrange multipliers
     * @param dx Search direction from QP
     * @param u multiplier from QP
     */
    public void update(final RealVector J,
                       final RealMatrix JE,
                       final RealMatrix JI,
                       final RealVector x,
                       final RealVector y,
                       final RealVector dx,
                       final RealVector u)
                       {

        this.J =J;
        this.JE=JE;
        this.JI=JI;
        this.x=x;
        this.y=y;
        this.dx=dx;
        this.u =u;
	this.penaltyGradient=gradient();
    }
     
    /**
     * get numbers of evaluation of Obejctive and Constraints
     * @return counter
     */
    int getCounter(){
    return this.evalCounter;
    }	
    
    
    /**
     * Penalty Gradient
     * @return penalty gradient
     */
    double getGradient(){
    return this.penaltyGradient;
    }	
    
    /**
     * Get Last Objective Evaluation;
     * @return penalty gradient
     */
    double getObjEval(){
    return this.objEval;
    }	
    
    /**
     * Get Last Inequality Constraints Evaluation;
     * @return penalty gradient
     */
    RealVector getIqEval(){
    return this.iqEval;
    }	
    
     /**
     * Get Last Equality Constraints Evaluation;
     * @return penalty gradient
     */
    RealVector getEqEval(){
    return this.eqEval;
    }	
    
    double getPenaltyEval(){
    return this.pEval;
    }	
	
    /**
     * Evaluate penalty function at x + alpha * dx.
     *
     * @param alpha Step length
     * @return penalty value
     */
    public double value(double alpha) {
        RealVector xAlpha = x.add(dx.mapMultiply(alpha));
        RealVector yAlpha=null;
        if(y.getDimension()>0)yAlpha = y.add(u.subtract(y).mapMultiply(alpha));
       
       
        objEval=this.objective.value(xAlpha);
        double penalty =objEval; 
       
        int me=0;
        int mi=0;
        if (eqConstraint != null) {
            me = eqConstraint.dimY();
            RealVector re = r.getSubVector(0, me);
            RealVector ye = yAlpha.getSubVector(0, me);
            eqEval=eqConstraint.value(xAlpha);
            RealVector g = eqEval.subtract(eqConstraint.getLowerBound());
            
            RealVector g2 = g.ebeMultiply(g);
            penalty -= ye.dotProduct(g) - 0.5 * re.dotProduct(g2);
        }

        if (iqConstraint != null) {
             mi = iqConstraint.dimY();
            RealVector ri = r.getSubVector(me, mi);
            RealVector yi = yAlpha.getSubVector(me, mi);
            
            RealVector yk = yAlpha.getSubVector(me, mi);
            //RealVector yk = y.getSubVector(me, mi);
            
            iqEval=iqConstraint.value(xAlpha);
            RealVector gk = iqEval.subtract(iqConstraint.getLowerBound());
            
            RealVector g = new ArrayRealVector(gk);
            RealVector mask = new ArrayRealVector(g.getDimension(), 1.0);

            for (int i = 0; i < gk.getDimension(); i++) {

                if (gk.getEntry(i) > (yk.getEntry(i) / ri.getEntry(i))) {

                    mask.setEntry(i, 0.0);

                    penalty -= 0.5 * yi.getEntry(i) * yi.getEntry(i) / ri.getEntry(i);
                }

            }

            RealVector g2 = g.ebeMultiply(g.ebeMultiply(mask));
            penalty -= yi.dotProduct(g.ebeMultiply(mask)) - 0.5 * ri.dotProduct(g2);
        }
        pEval=penalty;
        this.evalCounter++;
        return penalty;
    }

    /**
     * Get penalty gradient at current x.
     *
     * @return penalty gradient 
     */
    private double gradient() {
        if(y.getDimension()>0)
            return gradX().dotProduct(dx)+gradY().dotProduct(u.subtract(y));
        else
            return gradX().dotProduct(dx);    
    }
    
    
     public RealVector gradX() {
        RealVector partial =J;
//        RealVector x=this.x.copy();
//        RealVector y=this.y.copy();
        int me = 0;
        int mi;
       
        if (eqConstraint != null) {
            me = eqConstraint.dimY();
            RealVector re = r.getSubVector(0, me);
            RealVector ye = y.getSubVector(0, me);
           
//            RealVector ge = eqVal.subtract(eqConstraint.getLowerBound());
             RealVector ge = this.eqEval.subtract(eqConstraint.getLowerBound());
            RealMatrix jacob = JE;
            RealVector term=jacob.preMultiply(ye.subtract(ge.ebeMultiply(re)));
            partial=partial.subtract(term);
        }

        if (iqConstraint != null) {
            mi = iqConstraint.dimY();

            RealVector ri = r.getSubVector(me, mi);
         
            RealVector yi = y.getSubVector(me, mi);
//            RealVector gi = iqVal.subtract(iqConstraint.getLowerBound());
             RealVector gi = this.iqEval.subtract(iqConstraint.getLowerBound());
            RealMatrix jacob = JI;

            RealVector mask = new ArrayRealVector(mi, 1.0);
          
       

            for (int i = 0; i < gi.getDimension(); i++) {
                if (gi.getEntry(i) > yi.getEntry(i) / ri.getEntry(i)) mask.setEntry(i, 0.0);
            }        
  
            RealVector term=jacob.preMultiply((yi.subtract(gi.ebeMultiply(ri))).ebeMultiply(mask));
            partial=partial.subtract(term);
        }

        return partial;
    }
     
     
     public RealVector gradY() {

        int me = 0;
        int mi;
        RealVector partial = new ArrayRealVector(y.getDimension());
        if (eqConstraint != null) {
            me = eqConstraint.dimY();
//            RealVector g = eqVal.subtract(eqConstraint.getLowerBound());
             RealVector g = this.eqEval.subtract(eqConstraint.getLowerBound());
            partial.setSubVector(0, g.mapMultiply(-1.0));
        }

        if (iqConstraint != null) {
            mi = iqConstraint.dimY();

            RealVector ri = r.getSubVector(me, mi);
            RealVector yi = y.getSubVector(me, mi);
//            RealVector gi = iqVal.subtract(iqConstraint.getLowerBound());
            RealVector gi = this.iqEval.subtract(iqConstraint.getLowerBound());
           
            RealVector viri = new ArrayRealVector(mi, 0.0);
            for (int i = 0; i < gi.getDimension(); i++) {
              
                viri.setEntry(i,gi.getEntry(i) > yi.getEntry(i) / ri.getEntry(i)?-yi.getEntry(i)/ri.getEntry(i):-gi.getEntry(i));
            }

            //partial.setSubVector(me, (gi.ebeMultiply(mask).add(viri)).mapMultiply(-1.0));
            partial.setSubVector(me, viri);
        }

        return partial;
    }
    /**
     * Update Weight Vector Rj
     * called after QP solution before update the penalty function 
     * @param H hessina Matrix(updated after line search)
     * @@param y last estimate of multiplier(updated after line search)
     * @param dx direction of x provided by QP solution
     * @param u direction of y provided by QP solution
     * @param sigmaValue value of the additional variabile of QP solution
     * @param iterations current iteration
     */
     public void updateRj(RealMatrix H,RealVector y,RealVector dx, RealVector u,double sigmaValue,int iterations) { //r = updateRj(currentH,dx,y,u,r,sigm);
        //calculate sigma vector that dipend by iterations
        if (y.getDimension()==0)return;
        RealVector sigma = new ArrayRealVector(r.getDimension());
        for (int i = 0; i < sigma.getDimension(); i++) {
            final double appoggio = iterations / FastMath.sqrt(r.getEntry(i));
            sigma.setEntry(i, FastMath.min(1.0, appoggio));
        }

        int me = 0;
        int mi = 0;
        if (this.eqConstraint != null) {
            me = this.eqConstraint .dimY();
        }
        if (this.iqConstraint != null) {
            mi = this.iqConstraint.dimY();
        }

        RealVector sigmar = sigma.ebeMultiply(r);
        //(u-v)^2 or (ru-v)
        RealVector numerator = ((u.subtract(y)).ebeMultiply(u.subtract(y))).mapMultiply(2.0 * (mi + me));

        double denominator = dx.dotProduct(H.operate(dx)) * (1.0 - sigmaValue);
        RealVector r1 = new ArrayRealVector(r);
        if (this.eqConstraint != null) {
            for (int i = 0; i < me; i++) {
                r1.setEntry(i, FastMath.max(sigmar.getEntry(i), numerator.getEntry(i) / denominator));
            }
        }
        if (this.iqConstraint != null) {
            for (int i = 0; i < mi; i++) {
                r1.setEntry(me + i, FastMath.max(sigmar.getEntry(me + i), numerator.getEntry(me + i) / denominator));
            }
        }

        r.setSubVector(0,r1);
    }
   
     public void resetRj()
     {
         if(y.getDimension()>0)
            this.r.set(1.0);
     }

    RealVector getDx() {
       return  this.dx;
    }
}
