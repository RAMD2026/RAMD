package it.univaq.disim.architecturemodeling.launcher;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.Resource.Factory.Registry;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;


import org.eclipse.epsilon.common.parse.problem.ParseProblem;
import org.eclipse.epsilon.common.util.StringProperties;
import org.eclipse.epsilon.emc.emf.EmfModel;
import org.eclipse.epsilon.eol.IEolModule;
import org.eclipse.epsilon.eol.exceptions.EolRuntimeException;
import org.eclipse.epsilon.eol.exceptions.models.EolModelLoadingException;
import org.eclipse.epsilon.eol.execute.context.Variable;
import org.eclipse.epsilon.eol.models.IModel;
import org.eclipse.epsilon.eol.models.IRelativePathResolver;

public abstract class EpsilonStandaloneExample {
	
	protected IEolModule module;
	protected List<Variable> parameters = new ArrayList<Variable>();
	
	protected Object result;
	
	public abstract IEolModule createModule();
	
	public abstract String getSource() throws Exception;
	
	public abstract List<IModel> getModels() throws Exception;
	
	public void postProcess() {};
	
	public void preProcess() {};

	protected void registerResourceFactories() {
		Registry reg = Resource.Factory.Registry.INSTANCE;
		Map<String, Object> map = reg.getExtensionToFactoryMap();

		// đảm bảo EMF biết cách mở .ecore, .model, .xmi
		map.put("ecore", new XMIResourceFactoryImpl());
		map.put("model", new XMIResourceFactoryImpl());
		map.put("xmi",   new XMIResourceFactoryImpl());
	}

	public void execute() throws Exception {
		registerResourceFactories();
		module = createModule();
		module.parse(getFileURI(getSource()));
		
		if (module.getParseProblems().size() > 0) {
			System.err.println("Parse errors occured...");
			for (ParseProblem problem : module.getParseProblems()) {
				System.err.println(problem.toString());
			}
			return;
		}
		
		for (IModel model : getModels()) {
			module.getContext().getModelRepository().addModel(model);
		}
		
		for (Variable parameter : parameters) {
			module.getContext().getFrameStack().put(parameter);
		}
		
		preProcess();
		result = execute(module);
		postProcess();
		
		module.getContext().getModelRepository().dispose();
	}
	
	public List<Variable> getParameters() {
		return parameters;
	}
	
	protected Object execute(IEolModule module) 
			throws EolRuntimeException {
		return module.execute();
	}
	
	// protected EmfModel createEmfModel(String name, String model, 
	// 		String metamodel, boolean readOnLoad, boolean storeOnDisposal) 
	// 				throws EolModelLoadingException, URISyntaxException {
	// 	EmfModel emfModel = new EmfModel();
	// 	StringProperties properties = new StringProperties();
	// 	properties.put(EmfModel.PROPERTY_NAME, name);
	// 	properties.put(EmfModel.PROPERTY_FILE_BASED_METAMODEL_URI,
	// 			getFileURI(metamodel).toString());
	// 	properties.put(EmfModel.PROPERTY_MODEL_URI, 
	// 			getFileURI(model).toString());
		
	// 	properties.put(EmfModel.PROPERTY_READONLOAD, readOnLoad + "");
	// 	properties.put(EmfModel.PROPERTY_STOREONDISPOSAL, 
	// 			storeOnDisposal + "");
	// 	emfModel.load(properties, (IRelativePathResolver) null);
	// 	return emfModel;
	// }

	protected EmfModel createEmfModel(String name, String model, 
			String metamodel, boolean readOnLoad, boolean storeOnDisposal) 
					throws EolModelLoadingException, URISyntaxException {
		EmfModel emfModel = new EmfModel();
		StringProperties properties = new StringProperties();

		String modelUri     = getFileURI(model).toString();
		String metamodelUri = getFileURI(metamodel).toString();

		System.out.println("--------------------------------------------------");
		System.out.println("[createEmfModel] name       = " + name);
		System.out.println("[createEmfModel] modelPath  = " + model);
		System.out.println("[createEmfModel] metaPath   = " + metamodel);
		System.out.println("[createEmfModel] MODEL_URI  = " + modelUri);
		System.out.println("[createEmfModel] META_FILE  = " + metamodelUri);

		properties.put(EmfModel.PROPERTY_NAME, name);
		properties.put(EmfModel.PROPERTY_FILE_BASED_METAMODEL_URI, metamodelUri);
		properties.put(EmfModel.PROPERTY_MODEL_URI, modelUri);
		
		properties.put(EmfModel.PROPERTY_READONLOAD, readOnLoad + "");
		properties.put(EmfModel.PROPERTY_STOREONDISPOSAL, storeOnDisposal + "");

		try {
			emfModel.load(properties, (IRelativePathResolver) null);
			System.out.println("[createEmfModel] SUCCESS loading model '" + name + "'");
		}
		catch (Throwable t) {
			System.err.println("[createEmfModel] ERROR loading model '" + name + "'");
			System.err.println("  MODEL_URI     = " + modelUri);
			System.err.println("  META_FILE_URI = " + metamodelUri);
			t.printStackTrace();
			throw t;
		}

		return emfModel;
	}


	// protected EmfModel createEmfModelByURI(String name, String model, 
	// 		String metamodel, boolean readOnLoad, boolean storeOnDisposal) 
	// 				throws EolModelLoadingException, URISyntaxException {
	// 	EmfModel emfModel = new EmfModel();
	// 	StringProperties properties = new StringProperties();
	// 	properties.put(EmfModel.PROPERTY_NAME, name);
	// 	properties.put(EmfModel.PROPERTY_METAMODEL_URI, metamodel);
	// 	properties.put(EmfModel.PROPERTY_MODEL_URI, 
	// 			getFileURI(model).toString());
	// 	properties.put(EmfModel.PROPERTY_READONLOAD, readOnLoad + "");
	// 	properties.put(EmfModel.PROPERTY_STOREONDISPOSAL, 
	// 			storeOnDisposal + "");
	// 	emfModel.load(properties, (IRelativePathResolver) null);
	// 	return emfModel;
	// }

	protected EmfModel createEmfModelByURI(String name, String model, 
			String metamodel, boolean readOnLoad, boolean storeOnDisposal) 
					throws EolModelLoadingException, URISyntaxException {
		EmfModel emfModel = new EmfModel();
		StringProperties properties = new StringProperties();

		String modelUri = getFileURI(model).toString();

		System.out.println("--------------------------------------------------");
		System.out.println("[createEmfModelByURI] name        = " + name);
		System.out.println("[createEmfModelByURI] modelPath   = " + model);
		System.out.println("[createEmfModelByURI] MODEL_URI   = " + modelUri);
		System.out.println("[createEmfModelByURI] META_NS_URI = " + metamodel);

		properties.put(EmfModel.PROPERTY_NAME, name);
		properties.put(EmfModel.PROPERTY_METAMODEL_URI, metamodel);
		properties.put(EmfModel.PROPERTY_MODEL_URI, modelUri);
		properties.put(EmfModel.PROPERTY_READONLOAD, readOnLoad + "");
		properties.put(EmfModel.PROPERTY_STOREONDISPOSAL, storeOnDisposal + "");

		try {
			emfModel.load(properties, (IRelativePathResolver) null);
			System.out.println("[createEmfModelByURI] SUCCESS loading model '" + name + "'");
		}
		catch (Throwable t) {
			System.err.println("[createEmfModelByURI] ERROR loading model '" + name + "'");
			System.err.println("  MODEL_URI   = " + modelUri);
			System.err.println("  META_NS_URI = " + metamodel);
			t.printStackTrace();
			throw t;
		}

		return emfModel;
	}


	protected URI getFileURI(String fileName) throws URISyntaxException {

		File f = new File(fileName);
		if (f.exists()) {
			return f.toURI();
		}

		// 2) FALLBACK: CÁCH CŨ DÙNG getResource + bin→src (cho trường hợp chạy từ bin)
		URL binUrl = EpsilonStandaloneExample.class.getResource(
			fileName.startsWith("/") ? fileName : "/" + fileName
		);

		if (binUrl == null) {
			throw new RuntimeException("File not found (as filesystem path or classpath resource): " + fileName);
		}

		URI binUri = binUrl.toURI();
		URI uri = binUri;

		if (binUri.toString().contains("/bin/")) {
			// thay "bin" bằng "src" như logic gốc
			uri = URI.create(binUri.toString().replaceFirst("/bin/", "/src/"));
		}
		
		return uri;
	}
}
