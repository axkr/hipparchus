package org.hipparchus.optim.nonlinear.vector.constrained;

/**
 * Functional interface to allow lightweight debug logging without coupling.
 */
@FunctionalInterface
public interface DebugPrinter {
    void print(String message);
}
