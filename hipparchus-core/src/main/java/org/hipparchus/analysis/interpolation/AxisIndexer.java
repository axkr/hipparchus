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
package org.hipparchus.analysis.interpolation;

import org.hipparchus.exception.LocalizedCoreFormats;
import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.MathUtils;

import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Indexer for sampled data along a regular grid.
 * <p>
 * This class is intended to be check one axis of n-dimensional grid data.
 * </p>
 * <p>
 * A typical use case is loading data from a file with entries of the form:
 * </p>
 * <pre>
 *   x₁ y₁ valueA₁₁ valueB₁₁
 *   x₁ y₂ valueA₁₂ valueB₁₂
 *   ⋮
 *   x₁ yₙ valueA₁ₙ valueB₁ₙ
 *   x₂ y₁ valueA₂₁ valueB₂₁
 *   x₂ y₂ valueA₂₂ valueB₂₂
 *   ⋮
 *   x₂ yₙ valueA₂ₙ valueB₂ₙ
 *   x₃ yₙ valueA₃ₙ valueB₃ₙ
 *   ⋮
 *   xₘ yₙ valueAₘₙ valueBₘₙ
 * </pre>
 * <p>
 * In this case, the grid is two-dimensional, with x axis having m steps
 *  * ranging from x₁ to xₘ and y axis having n steps
 *  * ranging from y₁ to yₙ.
 * </p>
 * <p>
 * In this case, users could store the entries in some custom container
 * class {@code Entry} with getters {@code getX()}, {@code getY()}, {@code getValueA()}, and
 * {@code getValueB()}. Then the indexers for both x and y axes could be built as follows:
 * <pre>{@code
 *   AxisIndexer xIndexer = new AxisIndexer(List<Entry> data, Entry::getX(), tolerance);
 *   AxisIndexer yIndexer = new AxisIndexer(List<Entry> data, Entry::getY(), tolerance);
 * }</pre>
 * @since 4.1
 */
public class AxisIndexer {

    /** Minimum value. */
    private final double min;

    /** Maximum value. */
    private final double max;

    /** Number of points (including min and max). */
    private final int n;

    /** Step size. */
    private final double step;

    /** Simple constructor.
     * @param data grid data
     * @aram extractor for coordinate
     * @param tolerance tolerance below which extracted coordinates are considered equal
     */
    public <T> AxisIndexer(final List<T> data, ToDoubleFunction<T> extractor, final double tolerance) {

        // first pass, extract statistics
        double currentMin = Double.POSITIVE_INFINITY;
        double currentMax = Double.NEGATIVE_INFINITY;
        int    minCount   = 0;
        for (int i = 0; i < data.size(); i++) {
            final T t = data.get(i);
            final double coordinate = extractor.applyAsDouble(t);
            if (Double.isNaN(coordinate)) {
                throw new MathIllegalArgumentException(LocalizedCoreFormats.NAN_ELEMENT_AT_INDEX, i);
            }
            if (coordinate < currentMin - tolerance) {
                // this is a new minimum, we reset counter
                currentMin = coordinate;
                minCount   = 1;
            } else if (coordinate - currentMin < tolerance) {
                // we found again an already known minimum, update the count
                ++minCount;
            }
            currentMax = FastMath.max(currentMax, coordinate);
        }

        // store global indexing data
        this.min  = currentMin;
        this.max  = currentMax;
        this.n    = data.size() / minCount;
        this.step = (max - min) / (n - 1);

        // second pass, check grid data is regular
        final int[] count = new int[n];
        for (final T t : data) {
            count[index(extractor.applyAsDouble(t))]++;
        }
        for (int i = 0; i < n; i++) {
            if (count[i] != minCount) {
                throw new MathIllegalArgumentException(LocalizedCoreFormats.IRREGULAR_GRID, minCount, i, count[i]);
            }
        }

    }

    /** Get the minimum coordinate.
     * @return minimum coordinate
     */
    public double getMin() {
        return min;
    }

    /** Get the maximum coordinate.
     * @return maximum coordinate
     */
    public double getMax() {
        return max;
    }

    /** Get the number of different coordinates.
     * @return number of different coordinates (including {@link #getMin()} and {@link #getMax()})
     */
    public int getN() {
        return n;
    }

    /** Get the step size between two successive coordinates.
     * @return step size between two successive coordinates
     */
    public double getStep() {
        return step;
    }

    /** Get index corresponding to coordinate.
     * <p>
     * The method rounds to closest index, so it may be called with
     * {@code coordinate} slightly overshooting nominal range. If called
     * with {@code coordinate} between {@link #getMin()} - {@link #getStep()}/2
     * and {@link #getMin()} + {@link #getStep()}/2, it will return 0. If called
     * with {@code coordinate} between {@link #getMax()} - {@link #getStep()}/2
     * and {@link #getMax()} + {@link #getStep()}/2, it will return {@link #getN()} - 1.
     * If called outside of this extended range, it will throw an exception.
     * </p>
     * @param coordinate coordinate
     * @return index corresponding to coordinate (i.e. {@code 0} for {@code min}, {@code n-1} for {@code max})
     */
    public int index(final double coordinate) {
        final int index = (int) FastMath.rint((coordinate - min) / step);
        MathUtils.checkRangeInclusive(index, 0, n - 1);
        return index;
    }

    /** Get coordinates corresponding to index.
     * @param index index
     * @return coordinate (i.e. {@code min} for {@code 0}, {@code max} for {@code n-1})
     */
    public double coordinate(final int index) {
        MathUtils.checkRangeInclusive(index, 0, n - 1);
        // we count from the closest end to avoid numerical noise
        return index < n / 2 ?
               min + index * step :
               max - (n - 1 - index) * step;
    }

}
