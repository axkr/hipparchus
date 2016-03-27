/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.hipparchus.ode;

import java.util.ArrayList;
import java.util.Collection;

import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.exception.MathIllegalStateException;

/** Wrapper class enabling {@link FirstOrderDifferentialEquations basic simple}
 *  ODE instances to be used when processing {@link JacobianMatrices}.
 *
 */
class ParameterizedWrapper implements ParameterizedODE {

    /** Basic FODE without parameter. */
    private final FirstOrderDifferentialEquations fode;

    /** Simple constructor.
     * @param ode original first order differential equations
     */
    ParameterizedWrapper(final FirstOrderDifferentialEquations ode) {
        this.fode = ode;
    }

    /** Get the dimension of the underlying FODE.
     * @return dimension of the underlying FODE
     */
    public int getDimension() {
        return fode.getDimension();
    }

    /** Get the current time derivative of the state vector of the underlying FODE.
     * @param t current value of the independent <I>time</I> variable
     * @param y array containing the current value of the state vector
     * @param yDot placeholder array where to put the time derivative of the state vector
     * @exception MathIllegalStateException if the number of functions evaluations is exceeded
     * @exception MathIllegalArgumentException if arrays dimensions do not match equations settings
     */
    public void computeDerivatives(double t, double[] y, double[] yDot)
        throws MathIllegalArgumentException, MathIllegalStateException {
        fode.computeDerivatives(t, y, yDot);
    }

    /** {@inheritDoc} */
    @Override
    public Collection<String> getParametersNames() {
        return new ArrayList<String>();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSupported(String name) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public double getParameter(String name)
        throws MathIllegalArgumentException {
        if (!isSupported(name)) {
            throw new MathIllegalArgumentException(LocalizedODEFormats.UNKNOWN_PARAMETER, name);
        }
        return Double.NaN;
    }

    /** {@inheritDoc} */
    @Override
    public void setParameter(String name, double value) {
    }

}
