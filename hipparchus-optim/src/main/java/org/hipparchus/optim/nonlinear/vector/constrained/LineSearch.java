package org.hipparchus.optim.nonlinear.vector.constrained;

import java.util.LinkedList;
import java.util.Queue;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.Precision;

/**
 * Robust Line Search strategy
 *
 * This class manages monotone and non-monotone line search. Switching is
 * automatic after repeated failures of the monotone search.
 *
 * Typical parameter values used in Schittkowski's algorithms:
 * <ul>
 * <li>maxHistory = 5 to 10 (memory for non-monotone search)</li>
 * <li>sigma = 1e-4 or 1e-5 (Armijo sufficient decrease parameter)</li>
 * <li>beta = 0.5 (step reduction factor)</li>
 * <li>alphaMin (minimum allowed step length)</li>
 * <li>alphaMax = 1.0 (maximum allowed step length)</li>
 * <li>maxMonotoneFailures = 20 to 50 (failures before switching to
 * non-monotone)</li>
 * <li>maxBadSteps = 3 to 5 (allowed consecutive bad steps before Hessian
 * reset)</li>
 * </ul>
 */
public class LineSearch {

    private final int maxHistory;
    private final double sigma;
    private final double beta;
    private  double alphaMin;
    private final double alphaMax;
    private final int maxMonotoneFailures;
    private final int maxBadSteps;

    private final Queue<Double> history;
    private int searchCount;
    private double lastPenalty;
    private int monotoneFailures;
    private int badStepCount;
    private boolean nonMonotoneEnabled;
    private boolean badStepDetected;
    private boolean badStepFailed;

    public LineSearch(double eps, int maxHistory, double mu, double beta,
            int maxMonotoneFailures, int maxBadSteps) {
        this.maxHistory = maxHistory;
        this.sigma = mu;
        this.beta = beta;
        this.alphaMin = eps>1.0e-12?1.0e-12:eps;
        this.alphaMax = 1.0;
        this.maxMonotoneFailures = maxMonotoneFailures;
        this.maxBadSteps = maxBadSteps;
        this.history = new LinkedList<>();
        this.nonMonotoneEnabled = false;
        this.monotoneFailures = 0;
        this.badStepCount = 0;
        this.badStepDetected = false;
        this.badStepFailed = false;
    }

    public boolean isBadStepDetected() {
        return badStepDetected;
    }

    public boolean isBadStepFailed() {
        return badStepFailed;
    }

    public int getBadStepCount() {
        return badStepCount;
    }

    public int getIteration() {
        return searchCount;
    }

    public double getPenalty() {
        return lastPenalty;
    }

    public void resetBadStepCount() {
        badStepCount = 0;
        monotoneFailures = 0;

        badStepDetected = false;
        badStepFailed = false;
    }
    /**
     * Save penalty value when step is accepted for resusing in case of non monotone research
     * @param fx
     */
    public void updateHistory(double fx) {
        if (nonMonotoneEnabled) {
            history.add(fx);
            if (history.size() > maxHistory) {
                history.poll();
            }
        }
    }
   
    /**
     * Verify Armjo condition for accept step
     * @param fxNew
     * @param alpha
     * @param fxCurrent
     * @param directionalDeriv
     * @return true o false
     */
    public boolean acceptStep(double fxNew, double fxCurrent, double alpha, double directionalDeriv) {
        double ref = fxCurrent;
        if (nonMonotoneEnabled) {
            for (double v : history) {
                ref = Math.max(ref, v);
            }
        }
        // alfaPenalty - currentPenalty) > getSettings().getMu() * alpha * currentPenaltyGrad 
        return fxNew < ref + sigma * alpha * directionalDeriv;
    }
    /**
     * Mark Good Step if line search worked
     */
    public void markGoodStep() {
        nonMonotoneEnabled = false;
        monotoneFailures = 0;
        badStepCount = 0;
        badStepDetected = false;
        badStepFailed = false;
    }
    
    /**
     * Mark Bad Step if line search fail
     */
    private void markBadStep() {
        nonMonotoneEnabled = false;
        badStepCount++;
        badStepDetected = true;
        monotoneFailures = 0;
        if (badStepCount > maxBadSteps) {
            badStepFailed = true;
        }
    }
    /**
     * Update alpha qith quadratic curvature
     * @param alpha 
     * @param alpha penalty at current point 
     * @param fxNew penaly at candidate point x+dx*alpha 
     * @param directionalDeriv penalty gradinet
     * @return alpha
     */
    private double updateStepLength(double alpha, double fxCurrent, double fxNew, double directionalDeriv) {
        double numerator = 0.5 * alpha * alpha * directionalDeriv;
        double denominator = alpha * directionalDeriv - fxNew + fxCurrent;
        double alphaNew = alpha;
        if (Math.abs(denominator) > 1e-12) {
            double alphaStar = numerator / denominator;
            alphaNew = Math.max(alphaStar, alpha * beta);
        } else {
            alphaNew = alpha * beta;
        }

        return alphaNew;
    }

    /**
     * Main line search process: tries monotone first, then non-monotone.
     *
     * @param f penalty function
     * @return alpha
     */
    public double search(MeritFunctionL2 f) {
        searchCount=0;
        double fxCurrent = f.getPenaltyEval();
        alphaMin = Math.max(1e-12, 1e-6 / Math.max(1.0, f.getDx().getNorm()));
        //double fxCurrent=f.value(0);
        double alpha = 1.0;
        double fxNew;
        double directionalDeriv = f.getGradient();
        // Monotone Search
        while (alpha >=alphaMin) {
           
            fxNew = f.value(alpha);
            if (acceptStep(fxNew, fxCurrent, alpha, directionalDeriv)) {
                lastPenalty = fxNew;
                markGoodStep();
                updateHistory(fxNew);
                
                return alpha;
            }
            alpha = updateStepLength(alpha, fxCurrent, fxNew, directionalDeriv);
            monotoneFailures++;
            if (monotoneFailures >= maxMonotoneFailures) {
                nonMonotoneEnabled = true;
                break;
            }
             searchCount++;
        }
        
        // Non-Monotone Search
        monotoneFailures = 0;
        alpha = 1.0;

        while (alpha >=alphaMin) {
            
            fxNew = f.value(alpha);
            if (acceptStep(fxNew, fxCurrent, alpha, directionalDeriv)) {
                lastPenalty = fxNew;
                markGoodStep();
                updateHistory(fxNew);
                return alpha;
            }
            alpha = updateStepLength(alpha, fxCurrent, fxNew, directionalDeriv);
            monotoneFailures++;
            if (monotoneFailures >= maxMonotoneFailures) {
                nonMonotoneEnabled = false;
                break;
            }
            searchCount++;
        }
        //last trial before declare bad step
        fxNew = f.value(alphaMin);
        searchCount++;
        if (acceptStep(fxNew, fxCurrent, alphaMin, directionalDeriv))
        {
                lastPenalty = fxNew;
                markGoodStep();
                updateHistory(fxNew);
                return alpha;
        }
        // Both searches failed
        markBadStep();
        lastPenalty = fxNew;
        return alphaMin;

    }

}
