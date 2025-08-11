package org.hipparchus.optim.nonlinear.vector.constrained;


import org.hipparchus.util.Precision;

/**
 * Utility class for formatting SQP iteration logs with dynamic precision and aligned columns.
 */
public class SQPLogger {

    private int precision;
    private int width;
    private int lsWidth = 3; // LS column fixed to 2 digits + 1 space for safety
    private String headerFormat;
    private String rowFormat;
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
        String lsCol = String.format("%%-%ds", lsWidth);
        this.headerFormat = String.format(
            "[SQP] ITER %%2s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
            col, lsCol, col, col, col, col, col, col, col
        );

        String fld = String.format("%%" + width + "." + precision + "f");
        String intf = String.format("%%" + lsWidth + "d");
        this.rowFormat = String.format(
            "[SQP] ITER %%2d | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
            fld, intf, fld, fld, fld, fld, fld, fld, fld
        );
    }

    public void setDebugPrinter(DebugPrinter printer) {
        this.printer = printer;
    }

    public String header() {
        return String.format(headerFormat,
            "", "alpha", "LS", "dxNorm", "dx'Hdx", "KKT", "viol", "sigma", "penalty", "f(x)");
    }

    public String formatRow(int iter, double alpha, int lsCount,
                            double dxNorm, double dxHdx, double kkt,
                            double viol, double sigma, double penalty, double fx) {
        return String.format(rowFormat,
            iter, alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx);
    }

    public void logHeader() {
        if (printer != null) {
            printer.print(header());
        }
    }

    public void logRow(int iter, Object alpha, Object lsCount,
                       Object dxNorm, Object dxHdx, Object kkt,
                       Object viol, Object sigma, Object penalty, Object fx) {
        if (printer == null) return;
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[SQP] ITER %2d |", iter));
        Object[] fields = { alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx };
        int index = 0;
        for (Object field : fields) {
            String cell;
            if (field == null) {
                if (index == 1) {
                    cell = String.format(" %" + lsWidth + "s |", "");
                } else {
                    cell = String.format(" %" + width + "s |", "");
                }
            } else if (field instanceof Boolean) {
                if (index == 1) {
                    cell = String.format(" %" + lsWidth + "s |", field.toString());
                } else {
                    cell = String.format(" %" + width + "s |", field.toString());
                }
            } else if (index == 1 && field instanceof Number) { // LS column
                cell = String.format(" %" + lsWidth + "d |", ((Number) field).intValue());
            } else if (field instanceof Number) {
                cell = String.format(" %" + width + "." + precision + "f |", ((Number) field).doubleValue());
            } else {
                if (index == 1) {
                    cell = String.format(" %" + lsWidth + "s |", field.toString());
                } else {
                    cell = String.format(" %" + width + "s |", field.toString());
                }
            }
            sb.append(cell);
            index++;
        }
        printer.print(sb.toString());
    }

    public void logRow(int iter, double alpha, int lsCount,
                       double dxNorm, double dxHdx, double kkt,
                       double viol, double sigma, double penalty, double fx) {
        if (printer != null) {
            printer.print(formatRow(iter, alpha, lsCount, dxNorm, dxHdx, kkt, viol, sigma, penalty, fx));
        }
    }

    public SQPLogger defaultLogger() {
        return new SQPLogger(Precision.EPSILON);
    }
} 
