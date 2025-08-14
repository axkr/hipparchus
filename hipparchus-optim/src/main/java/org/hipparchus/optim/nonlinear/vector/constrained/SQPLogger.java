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

import org.hipparchus.util.Precision;

/**
 * Utility class for formatting SQP iteration logs with dynamic precision and aligned columns.
 */
public class SQPLogger {

    /** LS column fixed to 2 digits + 1 space for safety. */
    private static final int LS_WIDTH = 3;

    /** Field start. */
    private static final String FIELD_START = " %";

    /** Field continuation. */
    private static final String FIELD_CONTINUATION = "s |";

    /** Precision. */
    private int precision;

    /** Fields width. */
    private int width;

    /** Format of header line. */
    private String headerFormat;

    /** Format of row lines. */
    private String rowFormat;

    /** Debug printer. */
    private DebugPrinter printer;

    /**
     * Constructs a SQPLogger using {@link Precision#EPSILON} as default epsilon.
     */
    public SQPLogger() {
        setEps(Precision.EPSILON);
    }

    /**
     * Constructs a LogFormatter with custom epsilon.
     *
     * @param epsilon convergence threshold
     */
    public SQPLogger(double epsilon) {
        setEps(epsilon);
    }

    /**
     * Updates the formatter precision and formats based on a new epsilon value.
     *
     * @param epsilon convergence threshold
     */
    public void setEps(double epsilon) {
        this.precision = (int) Math.ceil(-Math.log10(epsilon)) + 2;
        this.width = precision + 7; // space for digits, sign, exponent, padding

        String col = String.format("%%-%ds", width);
        String lsCol = String.format("%%-%ds", LS_WIDTH);
        this.headerFormat = String.format(
            "[SQP] ITER %%2s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
            col, lsCol, col, col, col, col, col, col, col
        );

        String fld = String.format("%%" + width + "." + precision + "f");
        String intf = String.format("%%" + LS_WIDTH + "d");
        this.rowFormat = String.format(
            "[SQP] ITER %%2d | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
            fld, intf, fld, fld, fld, fld, fld, fld, fld
        );
    }

    /** Set debug printer.
     * @param printer debug printer
     */
    public void setDebugPrinter(DebugPrinter printer) {
        this.printer = printer;
    }

    /** Get header line.
     * @return header line
     */
    public String header() {
        return String.format(headerFormat,
            "", "alpha", "LS", "dxNorm", "dx'Hdx", "KKT", "viol", "sigma", "penalty", "f(x)");
    }

    /** Format one row.
     * @param iter     iteration number
     * @param alpha    step length
     * @param lsCount  line search iteration
     * @param dxNorm
     * @param dxHdx
     * @param kkt      Lagrangian norm
     * @param viol     constraints violations
     * @param sigma
     * @param penalty  penalty
     * @param fx       objective function evaluation
     * @return formatted row
     */
    public String formatRow(final int iter, final double alpha, final int lsCount,
                            final double dxNorm, final double dxHdx, final double kkt,
                            final double viol, final double sigma, final double penalty, final double fx) {
        return String.format(rowFormat,
                             iter, alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx);
    }

    /** Log header.
     */
    public void logHeader() {
        if (printer != null) {
            printer.print(header());
        }
    }

    /** Log one row.
     * @param iter     iteration number
     * @param alpha    step length
     * @param lsCount  line search iteration
     * @param dxNorm
     * @param dxHdx
     * @param kkt      Lagrangian norm
     * @param viol     constraints violations
     * @param sigma
     * @param penalty  penalty
     * @param fx       objective function evaluation
     */
    public void logRow(final int iter, final Object alpha, final Object lsCount,
                       final Object dxNorm, final Object dxHdx, final Object kkt,
                       final Object viol, final Object sigma, final Object penalty, final Object fx) {
        if (printer == null) {
            return;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[SQP] ITER %2d |", iter));
        Object[] fields = { alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx };
        int index = 0;
        for (Object field : fields) {
            String cell;
            if (field == null) {
                if (index == 1) {
                    cell = String.format(FIELD_START + LS_WIDTH + FIELD_CONTINUATION, "");
                } else {
                    cell = String.format(FIELD_START + width + FIELD_CONTINUATION, "");
                }
            } else if (field instanceof Boolean) {
                if (index == 1) {
                    cell = String.format(FIELD_START + LS_WIDTH + FIELD_CONTINUATION, field);
                } else {
                    cell = String.format(FIELD_START + width + FIELD_CONTINUATION, field);
                }
            } else if (index == 1 && field instanceof Number) { // LS column
                cell = String.format(FIELD_START + LS_WIDTH + "d |", ((Number) field).intValue());
            } else if (field instanceof Number) {
                cell = String.format(FIELD_START + width + "." + precision + "f |", ((Number) field).doubleValue());
            } else {
                if (index == 1) {
                    cell = String.format(FIELD_START + LS_WIDTH + FIELD_CONTINUATION, field);
                } else {
                    cell = String.format(FIELD_START + width + FIELD_CONTINUATION, field);
                }
            }
            sb.append(cell);
            index++;
        }
        printer.print(sb.toString());
    }

    /** Log one row.
     * @param iter     iteration number
     * @param alpha    step length
     * @param lsCount  line search iteration
     * @param dxNorm
     * @param dxHdx
     * @param kkt      Lagrangian norm
     * @param viol     constraints violations
     * @param sigma
     * @param penalty  penalty
     * @param fx       objective function evaluation
     */
    public void logRow(int iter, double alpha, int lsCount,
                       double dxNorm, double dxHdx, double kkt,
                       double viol, double sigma, double penalty, double fx) {
        if (printer != null) {
            printer.print(formatRow(iter, alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx));
        }
    }

    /** Get default logger.
     * @return default logger
     */
    public SQPLogger defaultLogger() {
        return new SQPLogger(Precision.EPSILON);
    }

}
