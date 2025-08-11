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
import org.hipparchus.linear.CholeskyDecomposition;
import org.hipparchus.linear.MatrixUtils;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.Precision;

/**
 * BFGS Hessian updater with dynamic damping and robustness improvements.
 * <p>
 * Manages Hessian updates for SQP solvers by:
 * <ul>
 *   <li>Checking curvature condition</li>
 *   <li>Applying dynamic damping if necessary</li>
 *   <li>Skipping update if curvature still fails after damping</li>
 *   <li>Soft regularization of diagonal entries on repeated failures</li>
 *   <li>Automatic Hessian reset after configurable failures</li>
 * </ul>
 * </p>
 *
 * @since 3.1
 */
public class BFGSUpdater {

    /** Damping factor. */
    private final double GAMMA = 0.2;

    /** Regularization factor for diagonal of Hessian. */
    private final double regFactor;

    /** Maximum consecutive failures allowed before reset. */
    private final int maxFail;

    /** Stored initial Hessian for resets. */
    private final RealMatrix initialH;

    /** Current Cholesky factor L such that H = L·Lᵀ. */
    private RealMatrix L;

    /** Failure counter for damping/regularization. */
    private int failCount;

    /**
     * Creates a new updater.
     *
     * @param initialHess initial positive‐definite Hessian matrix
     * @param regFactor regularization factor to add on diagonal
     * @param maxFail maximum number of failures before Hessian reset
     */
    public BFGSUpdater(RealMatrix initialHess, double regFactor, int maxFail) {
        this.initialH = new Array2DRowRealMatrix(initialHess.getData());
        //this.regFactor = regFactor;
        this.regFactor = Precision.EPSILON;
        this.maxFail = maxFail;
        resetHessian();
    }

    /**
     * Returns the current Hessian matrix H = L·Lᵀ.
     *
     * @return current Hessian
     */
    public RealMatrix getHessian() {
        return L.multiplyTransposed(L);
    }

    /**
     * Returns the inverse of the current L factor.
     *
     * @return inverse of lower‐triangular L
     */
    public RealMatrix getInvL() {
        return inverseLowerTriangular(L);
    }

    /**
     * Updates the Hessian approximation using the BFGS formula.
     * <p>
     * If curvature condition fails, applies damping or regularization.
     *</p>
     *
     * @param s displacement vector (x_{k+1} − x_k)
     * @param y1 gradient difference (∇f_{k+1} − ∇f_k)
     */
    public void update(RealVector s, RealVector y1) {
        RealVector y = damp(s, y1);
        if (y == null) {
            return;
        }
        // Attempt rank‐one BFGS update; regularize on failure
        rankOneUpdate(s, y);
        failCount = 0;
    }

    /**
     * Applies soft regularization by adding regFactor to diagonal of H.
     */
    public void regularize() {
        double minDiagSqr = Double.POSITIVE_INFINITY;
         for (int i = 0; i < L.getRowDimension(); i++) {
           double lii = L.getEntry(i, i);
           minDiagSqr = Math.min(minDiagSqr, lii * lii);
        }
        double lambda = FastMath.min(0.0,minDiagSqr);
        RealMatrix A = L.multiplyTransposed(L);
        
        RealMatrix scaledI = MatrixUtils.createRealIdentityMatrix(A.getRowDimension()).scalarMultiply(regFactor-lambda);
        A = A.add(scaledI);
        RealMatrix Lp = new CholeskyDecomposition(A).getL();
       
        L.setSubMatrix(Lp.getData(), 0, 0);
    }

    /**
     * Resets the Hessian approximation to its initial value.
     */
    public void resetHessian() {
        CholeskyDecomposition ch = new CholeskyDecomposition(initialH);
        L = ch.getL();
        failCount = 0;
    }
    
    /**
     * Resets the Hessian approximation to its initial value.
     */
    public void resetHessian(double gamma) {
        CholeskyDecomposition ch = new CholeskyDecomposition(MatrixUtils.createRealIdentityMatrix(L.getRowDimension()).scalarMultiply(gamma));
        L = ch.getL();
        failCount = 0;
    }

    /**
     * Applies dynamic damping to y to satisfy curvature condition sᵀy ≥ γ sᵀHs.
     *
     * @param s search direction
     * @param y1 raw gradient difference
     * @return damped y, or null if update should be skipped
     */
    public RealVector damp(RealVector s, RealVector y1) {
        RealVector y = new ArrayRealVector(y1);
        double sty = s.dotProduct(y1);
        RealVector Hs = getHessian().operate(s);
        double sHs = s.dotProduct(Hs);
        if(sty<=Precision.EPSILON) return null;
        if (sty < GAMMA * sHs) {
            double phi = (1.0 - GAMMA) * sHs / (sHs - sty);
            // clamp phi to [0,1]
            phi = FastMath.max(0.0, FastMath.min(1.0, phi));
            y = y1.mapMultiply(1.0 - phi).add(Hs.mapMultiply(phi));
            sty = s.dotProduct(y);
            if (sty < GAMMA * sHs) {
                failCount++;
                return null;
            }
        }
        return y;
    }

    /**
     * Computes the inverse of a lower‐triangular matrix via forward substitution.
     *
     * @param L lower‐triangular matrix
     * @return inverse of L
     */
    private RealMatrix inverseLowerTriangular(RealMatrix L) {
        int n = L.getRowDimension();
        RealMatrix Linv = MatrixUtils.createRealMatrix(n, n);
        for (int i = 0; i < n; i++) {
            RealVector e = new ArrayRealVector(n);
            e.setEntry(i, 1.0);
            MatrixUtils.solveLowerTriangularSystem(L, e);
            Linv.setColumnVector(i, e);
        }
        return Linv;
    }

    /**
     * Performs a BFGS rank‐one update on L.
     *
     * @param s displacement vector
     * @param y gradient difference vector
     * @return true if update succeeded, false otherwise
     */
    private boolean rankOneUpdate(RealVector s, RealVector y) {
        RealMatrix Lcopy = new Array2DRowRealMatrix(L.getData());
        RealVector Hs = L.operate(L.preMultiply(s));
        double rho = 1.0 / FastMath.sqrt(s.dotProduct(y));
        double theta = 1.0 / FastMath.sqrt(s.dotProduct(Hs));
        RealVector v = y.mapMultiply(rho);
        RealVector w = Hs.mapMultiply(theta);
//        if (!cholupdateLower(L, v, +1)) {
//            //regularize();
//            return false;
//        }
             cholupdateLower(L, v, +1) ;
        
        if (!cholupdateLower(L, w, -1)) {
            //try to regularize
            
                
                   L.setSubMatrix(Lcopy.getData(), 0, 0);
            
            
        }
        return true;
    }

    /**
     * Performs a rank‐one Cholesky update/downdate on L.
     * <p>
     * Updates L such that A'=A+σu uᵀ or A'=A−u uᵀ, without refactorization.
     * </p>
     *
     * @param L lower‐triangular factor (overwritten)
     * @param u update vector
     * @param sigma +1 for update, -1 for downdate
     * @return true if resulting matrix remains PD, false otherwise
     */
    private boolean cholupdateLower(RealMatrix L, RealVector u, int sigma) {
        int n = u.getDimension();
        RealVector temp = new ArrayRealVector(u);
        for (int i = 0; i < n; i++) {
            double lii = L.getEntry(i, i);
            double ui = temp.getEntry(i);
            double r2 = lii * lii + sigma * ui * ui;
            if (r2 < regFactor) {
                return false;
            }
            double r = Math.sqrt(r2);
            double c = r / lii;
            double s = ui / lii;
            L.setEntry(i, i, r);
            for (int j = i + 1; j < n; j++) {
                double lji = L.getEntry(j, i);
                double uj = temp.getEntry(j);
                double newLji = (lji + sigma * s * uj) / c;
                double newUj = c * uj - s * newLji;
                L.setEntry(j, i, newLji);
                temp.setEntry(j, newUj);
            }
        }
        return true;
    }
    
    /**
 * Performs the rank-one update on the LDL decomposition.
 * If the diagonal elements of D become too small, returns false.
 * 
 * @param L the lower triangular matrix
 * @param D the diagonal matrix
 * @param u the update vector
 * @param sigma +1 for update, -1 for downdate
 * @return true if the matrix remains positive definite, false otherwise
 */
private boolean ldlUpdateRankOne(RealMatrix L, RealMatrix D, RealVector u, int sigma) {
    int n = u.getDimension();
    RealVector temp = new ArrayRealVector(u);  // Copy of u to avoid in-place modifications

    // Loop for updating the matrix L and the diagonal of D
    for (int i = 0; i < n; i++) {
        double lii = L.getEntry(i, i);  // Diagonal element of L
        double ui = temp.getEntry(i);   // Corresponding element of u
        double r2 = lii * lii + sigma * ui * ui;  // Calculation of r^2

        double r = Math.sqrt(r2);  // Calculate the square root of r^2
        double c = r / lii;        // Calculate the coefficient c
        double s = ui / lii;       // Calculate the coefficient s

        // Update the diagonal element of L
        L.setEntry(i, i, r);

        // Update the diagonal of D
        double dii = D.getEntry(i, i);
        double updatedDii = dii + sigma * r * s * s; // Update of D

        // Check if the diagonal element of D becomes too small
        if (updatedDii < regFactor) {
            return false;  // If the value is too small, return false
        }

        // Update the diagonal of D
        D.setEntry(i, i, updatedDii);

        // Update the elements below the diagonal of L
        for (int j = i + 1; j < n; j++) {
            double lij = L.getEntry(j, i);  // Element L_{ji}
            double uj = temp.getEntry(j);   // Corresponding element of u
            double newLji = (lij + sigma * s * uj) / c;  // Calculate the new value of L_{ji}
            double newUj = c * uj - s * newLji;  // Calculate the new value of u_j
            L.setEntry(j, i, newLji);    // Update L_{ji}
            temp.setEntry(j, newUj);     // Update u_j
        }
    }

    return true;  // Returns true if the update was executed correctly
}

    
}
