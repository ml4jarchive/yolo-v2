/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.models.yolov2.impl;

import java.util.Arrays;

import org.ml4j.Matrix;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;

/**
 * @author Michael Lavelle
 */
public class DefaultUntrainedYOLOv2WeightsLoader implements YOLOv2WeightsLoader {

	@Override
	public WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		if (width == 1 && height == 1) {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		} else {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		}
	}

	@Override
	public WeightsMatrix getBatchNormLayerWeights(String name, int inputDepth) {
		return new WeightsMatrixImpl(null,
				new WeightsFormatImpl(Arrays.asList(
						Dimension.INPUT_DEPTH), 
						Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	}

	@Override
	public Matrix getBatchNormLayerMovingVariance(String name, int inputDepth) {
		return null;
	}

	@Override
	public Matrix getBatchNormLayerMovingMean(String name, int inputDepth) {
		return null;
	}

	@Override
	public BiasMatrix getBatchNormLayerBias(String name, int inputDepth) {
		return null;
	}

	@Override
	public BiasMatrix getConvolutionalLayerBiases(String name, int outputDepth) {
		return null;
	}

}
