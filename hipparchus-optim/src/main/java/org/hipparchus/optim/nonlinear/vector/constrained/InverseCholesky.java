/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.hipparchus.optim.nonlinear.vector.constrained;

import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.OptimizationData;

/**
 *
 * @author rocca
 */
public class InverseCholesky implements OptimizationData{
    private final RealMatrix invL;
   
    public InverseCholesky(RealMatrix inverseL)
    {
       this.invL=inverseL; 
    }
    public RealMatrix get()
    {
        return this.invL;
    }
}
