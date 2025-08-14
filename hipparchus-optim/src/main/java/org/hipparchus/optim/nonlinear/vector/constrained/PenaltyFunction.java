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

import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.ArrayRealVector;


/**
 * Penalty Function manager inspired by Schittkowski (NLPQL, NLPQLP).
 * <p>
 * This class computes the penalty function and its gradient, combining:
 * </p>
 * <ul>
 *   <li>The objective function</li>
 *   <li>Equality constraints</li>
 *   <li>Inequality constraints</li>
 * </ul>
 * <p>
 * Typical usage:
 * </p>
 * <ul>
 *   <li>Call update(...) before line search or Hessian update</li>
 *   <li>Use value(alpha) to evaluate penalty at x + alpha * dx</li>
 *   <li>Use gradient() to retrieve penalty gradient at current point</li>
 *  </ul>
 */
public abstract class PenaltyFunction {

    /** Objective function. */
    private final TwiceDifferentiableFunction objective;

    /** Equality constraints (may be null). */
    private final Constraint eqConstraint;

    /** Inequality constraints (may be null). */
    private final Constraint iqConstraint;

    /** current point. */
    private RealVector x;

    /** Lagrange multipliers. */
    private RealVector y;

    /** Search direction. */
    private RealVector dx;

    /** Change in multipliers. */
    private RealVector dy;

    /** Unit vector. */
    private final RealVector r;

    /** Gradient of the objective at current point. */
    private RealVector J;

    /** penalty gradient. */
    private double penaltyGradient;

    /** Objective function evaluation. */
    private double objEval;

    /** Equality evaluation. */
    private RealVector eqEval;

    /** Inequality evaluation. */
    private RealVector iqEval;

    /** Penalty evaluation. */
    private double pEval;

    /** Gradient of equality. */
    private RealMatrix JE;

    /** Gradient of inequality. */
    private RealMatrix JI;

    /** Simple constructor.
     * @param objective Objective function
     * @param eqConstraint Equality constraint (may be null)
     * @param iqConstraint Inequality constraint (may be null)
     * @param x current point
     */
    public PenaltyFunction(final TwiceDifferentiableFunction objective,
                           final Constraint eqConstraint,
                           final Constraint iqConstraint,
                           final RealVector x) {
        this.objective = objective;
        this.eqConstraint = eqConstraint;
        this.iqConstraint = iqConstraint;
        this.x = x.copy();
        int m = 0;
        if (this.eqConstraint != null) {
            m = m + this.eqConstraint.dimY();
        }
        if (this.iqConstraint != null) {
            m = m + this.iqConstraint.dimY();
        }
        this.dx = new ArrayRealVector(x.getDimension());
        this.y  = new ArrayRealVector(m);
        this.dy = new ArrayRealVector(m);
        this.r  = new ArrayRealVector(m, 1.0);
        //this evaluates objective function constraints function and penalty
        this.value(0);
    }

    /** Update internal parameters for next penalty computation.
     * @param J Gradient of objective at current x
     * @param JE Gradient of equality x
     * @param JI Gradient of inequality at current x
     * @param x Current iterate
     * @param y Lagrange multipliers
     * @param dx Search direction
     * @param dy Change in multipliers
     */
    public void update(final RealVector J, final RealMatrix JE, final RealMatrix JI,
                       final RealVector x, final RealVector y,
                       final RealVector dx, final RealVector dy) {

        this.J  = J;
        this.JE = JE;
        this.JI = JI;
        this.x  = x;
        this.y  = y;
        this.dx = dx;
        this.dy = dy;
        this.penaltyGradient = gradient();
    }

    /** Get penalty Gradient.
     * @return penalty gradient
     */
    public double getGradient() {
        return penaltyGradient;
    }

    /** Get last objective evaluation.
     * @return last objective evaluation
     */
    public double getObjectiveEval() {
        return objEval;
    }

    /** Get last inequality evaluation.
     * @return last inequality evaluation
     */
    public RealVector getInequalityEval() {
        return iqEval;
    }

    /** Get last equality evaluation.
     * @return last equality evaluation
     */
    public RealVector getEqualityEval() {
        return eqEval;
    }

    /** Get last penalty evaluation.
     * @return last penalty evaluation
     */
    public double getPenaltyEval() {
        return pEval;
    }

    /** Evaluate penalty function at x + alpha * dx.
     * @param alpha Step length
     * @return penalty value
     */
    public double value(double alpha) {
        RealVector xAlpha = x.add(dx.mapMultiply(alpha));
        RealVector yAlpha = y.add(dy.subtract(y).mapMultiply(alpha));


        objEval = this.objective.value(xAlpha);
        double penalty =objEval;

        int me = 0;
        if (eqConstraint != null) {
            me = eqConstraint.dimY();
            RealVector re = r.getSubVector(0, me);
            RealVector ye = yAlpha.getSubVector(0, me);
            eqEval = eqConstraint.value(xAlpha);
            RealVector g = eqEval.subtract(eqConstraint.getLowerBound());

            RealVector g2 = g.ebeMultiply(g);
            penalty -= ye.dotProduct(g) - 0.5 * re.dotProduct(g2);
        }

        int mi = 0;
        if (iqConstraint != null) {
            mi = iqConstraint.dimY();
            RealVector ri = r.getSubVector(me, mi);
            RealVector yi = yAlpha.getSubVector(me, mi);
            RealVector yk = yAlpha.getSubVector(me, mi);

            iqEval = iqConstraint.value(xAlpha);
            RealVector gk = iqEval.subtract(iqConstraint.getLowerBound());

            RealVector g = gk.copy();
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
        pEval = penalty;
        return penalty;
    }

    /** Get penalty gradient at current x.
     * @return penalty gradient
     */
    private double gradient() {
        return gradX().dotProduct(dx) + gradY().dotProduct(dy.subtract(y));
    }

    public RealVector gradX() {
        RealVector partial = J.copy();
        int        me      = 0;
        int        mi;

        if (eqConstraint != null) {
            me = eqConstraint.dimY();
            RealVector re = r.getSubVector(0, me);
            RealVector ye = y.getSubVector(0, me);

            RealVector ge    = this.eqEval.subtract(eqConstraint.getLowerBound());
            RealMatrix jacob = JE;
            RealVector term  = jacob.transpose().operate(ye.subtract(ge.ebeMultiply(re)));
            partial = partial.subtract(term);
        }

        if (iqConstraint != null) {
            mi = iqConstraint.dimY();

            RealVector ri    = r.getSubVector(me, mi);
            RealVector yi    = y.getSubVector(me, mi);
            RealVector gi    = this.iqEval.subtract(iqConstraint.getLowerBound());
            RealMatrix jacob = JI;
            RealVector mask  = new ArrayRealVector(mi, 1.0);

            for (int i = 0; i < gi.getDimension(); i++) {
                if (gi.getEntry(i) > yi.getEntry(i) / ri.getEntry(i)) {
                    mask.setEntry(i, 0.0);
                }
            }

            RealVector term = jacob.transpose().operate((yi.subtract(gi.ebeMultiply(ri))).ebeMultiply(mask));
            partial = partial.subtract(term);
        }

        return partial;
    }

    public RealVector gradY() {

        int me = 0;
        int mi;
        RealVector partial = new ArrayRealVector(y.getDimension());
        if (eqConstraint != null) {
            me = eqConstraint.dimY();
             RealVector g = this.eqEval.subtract(eqConstraint.getLowerBound());
            partial.setSubVector(0, g.mapMultiply(-1.0));
        }

        if (iqConstraint != null) {
            mi = iqConstraint.dimY();

            RealVector ri = r.getSubVector(me, mi);
            RealVector yi = y.getSubVector(me, mi);
            RealVector gi = this.iqEval.subtract(iqConstraint.getLowerBound());

            RealVector viri = new ArrayRealVector(mi, 0.0);
            for (int i = 0; i < gi.getDimension(); i++) {
                viri.setEntry(i,
                              gi.getEntry(i) > yi.getEntry(i) / ri.getEntry(i) ?
                              -yi.getEntry(i) / ri.getEntry(i) :
                              -gi.getEntry(i));
            }

            partial.setSubVector(me, viri);
        }

        return partial;

    }

}
