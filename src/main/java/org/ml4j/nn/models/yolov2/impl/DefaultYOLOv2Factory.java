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

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2Definition;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.models.yolov2.YOLOv2Factory;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.ml4j.nn.sessions.factories.SessionFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default factory for an YOLO v2 Network.
 * 
 */
public class DefaultYOLOv2Factory implements YOLOv2Factory {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultYOLOv2Factory.class);

	private SessionFactory<DefaultChainableDirectedComponent<?, ?>> sessionFactory;

	private SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory;

	private YOLOv2WeightsLoader weightsLoader;

	private YOLOv2Labels labels;

	/**
	 * Creates the default pre-trained YOLO v2 Networks
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param classLoader
	 * @throws IOException
	 */
	public DefaultYOLOv2Factory(SessionFactory<DefaultChainableDirectedComponent<?, ?>> sessionFactory,
			MatrixFactory matrixFactory,
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			ClassLoader classLoader) throws IOException {
		this(sessionFactory, supervisedFeedForwardNeuralNetworkFactory,
				new PretrainedYOLOv2WeightsLoaderImpl(classLoader, matrixFactory),
				new DefaultYOLOv2Labels(classLoader));
	}

	/**
	 * Creates YOLO v2 Network with custom weights and labels
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param weightsLoader
	 * @param labels
	 */
	public DefaultYOLOv2Factory(SessionFactory<DefaultChainableDirectedComponent<?, ?>> sessionFactory,
			SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			YOLOv2WeightsLoader weightsLoader, YOLOv2Labels labels) {
		this.sessionFactory = sessionFactory;
		this.supervisedFeedForwardNeuralNetworkFactory = supervisedFeedForwardNeuralNetworkFactory;
		this.weightsLoader = weightsLoader;
		this.labels = labels;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createYoloV2(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {

		LOGGER.info("Creating Yolo V2 Network...");

		// Create a YOLOv2Definition from neural-network-architectures, initialising with the weights loader.
		YOLOv2Definition inceptionV4Definition = new YOLOv2Definition(weightsLoader);

		// Create a graph builder for the YOLOv2Definition and Training Context.
		// Add a linear activation function, as NeuralNetwork instances must end with an activation function.
		InitialComponents3DGraphBuilder<DefaultChainableDirectedComponent<?, ?>> graphBuilder = sessionFactory
				.createSession(trainingContext.getDirectedComponentsContext()).buildComponentGraph()
				.startWith(inceptionV4Definition).withActivationFunction("output", 
						ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR), new ActivationFunctionProperties());

		// Create the component graph from the definition and graph builder, and wrap
		// with a supervised feed forward neural network.
		return supervisedFeedForwardNeuralNetworkFactory
				.createSupervisedFeedForwardNeuralNetwork("inceptionv4_network", graphBuilder.getComponents());
	}

	@Override
	public YOLOv2Labels createYoloV2Labels() throws IOException {
		return labels;
	}
}
