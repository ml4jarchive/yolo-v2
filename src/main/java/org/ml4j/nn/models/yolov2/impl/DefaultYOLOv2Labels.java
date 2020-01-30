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
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.ml4j.nn.models.yolov2.YOLOv2Labels;

/**
 * The labels for the default pretrained YOLO v2 network
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultYOLOv2Labels implements YOLOv2Labels {

	private Map<Integer, String> classificationNamesByIndex;

	public DefaultYOLOv2Labels(ClassLoader classLoader) throws IOException {
		classificationNamesByIndex = new HashMap<>();
		try (InputStream is = classLoader.getResourceAsStream("coco_classes.txt")) {
			int index = 0;
			try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
				while (scanner.useDelimiter("\n").hasNext()) {
					classificationNamesByIndex.put(index, scanner.useDelimiter("\n").next());
					index++;
				}
			}
			if (index != 80) {
				throw new IllegalStateException("YOLO v2 Classification Names Load Error");
			}
		}
	}

	public String getLabel(int labelIndex) {
		String classificationName = classificationNamesByIndex.get(labelIndex);
		if (classificationName == null) {
			throw new IllegalArgumentException("Index of:" + labelIndex + " is out of range");
		}
		return classificationName;
	}
}
