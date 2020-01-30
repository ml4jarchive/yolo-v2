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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Michael Lavelle
 */
public class PretrainedYOLOv2WeightsLoaderImpl implements YOLOv2WeightsLoader {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultYOLOv2Factory.class);

	private MatrixFactory matrixFactory;
	private ClassLoader classLoader;
	private long uid;

	public PretrainedYOLOv2WeightsLoaderImpl(ClassLoader classLoader, MatrixFactory matrixFactory) {
		this.uid = ObjectStreamClass.lookup(float[].class).getSerialVersionUID();
		this.classLoader = classLoader;
		this.matrixFactory = matrixFactory;
	}

	public static PretrainedYOLOv2WeightsLoaderImpl getLoader(MatrixFactory matrixFactory,
			ClassLoader classLoader) {
		return new PretrainedYOLOv2WeightsLoaderImpl(classLoader, matrixFactory);
	}

	private float[] deserializeWeights(String name) {
		LOGGER.debug("Derializing weights:" + name);
		try {
			return deserialize(float[].class, "yolov2javaweights", uid, name);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public Matrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		float[] weights = deserializeWeights(name);
		// is height, width, inputDepth * outputDepth
		// want outputDepth * inputDepth, height, width.
		Matrix weightsMatrix1 =  matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights);

		
		// This is outputDepth * height, width, inputDepth
		Matrix weightsMatrix =  matrixFactory.createMatrixFromRowsByRowsArray(width * height * inputDepth, outputDepth, weights).transpose();
		Matrix outputWeights = matrixFactory.createMatrix(weightsMatrix.getRows(), weightsMatrix.getColumns());
		for (int r = 0; r < weightsMatrix.getRows(); r++) {
			
			// height, width, inputdepth
			Matrix rowData = weightsMatrix.getRow(r);
			rowData.asEditableMatrix().reshape(inputDepth, height * width);
			Matrix rowData2 = matrixFactory.createMatrix(rowData.getRows(), rowData.getColumns());
			for (int i = 0; i < rowData.getRows(); i++) {
				Matrix d = rowData.getRow(i);
				d.asEditableMatrix().reshape(height, width);
				d = d.transpose();
				rowData2.asEditableMatrix().putRow(i, d);
			}


			outputWeights.asEditableMatrix().putRow(r, rowData2);
			
		}
	
	
		
		return weightsMatrix1;
	}

	public Matrix getBatchNormLayerWeights(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}

	@SuppressWarnings("unchecked")
	public <S extends Serializable> S deserialize(Class<S> clazz, String path, long uid, String id)
			throws IOException, ClassNotFoundException {

		if (classLoader == null) {
			try (InputStream is = new FileInputStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}
		} else {
			try (InputStream is = classLoader
					.getResourceAsStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}

		}
	}

	@Override
	public Matrix getBatchNormLayerMovingVariance(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}

	@Override
	public Matrix getBatchNormLayerMovingMean(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}

	@Override
	public Matrix getBatchNormLayerBias(String name, int inputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(inputDepth, 1, weights);
	}

	@Override
	public Matrix getConvolutionalLayerBiases(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return matrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights);
	}
}
