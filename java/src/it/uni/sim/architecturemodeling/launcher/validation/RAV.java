package it.uni.sim.architecturemodeling.launcher.validation;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.Resource.Factory.Registry;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import org.eclipse.epsilon.eol.IEolModule;
import org.eclipse.epsilon.eol.exceptions.EolRuntimeException;
import org.eclipse.epsilon.eol.models.IModel;
import org.eclipse.epsilon.evl.EvlModule;
import org.eclipse.epsilon.evl.execute.FixInstance;
import org.eclipse.epsilon.evl.execute.UnsatisfiedConstraint;

import it.uni.sim.architecturemodeling.launcher.EpsilonStandaloneExample;

public class RAV extends EpsilonStandaloneExample {

    private static String modelPath;
    private static String metamodelPath;
    private static String referenceArchitecturePath;

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java ... RAV <modelPath> <metamodelPath> <referenceArchitecturePath>");
            System.err.println(
                "Example: java -cp bin:lib/* " +
                "it.uni.sim.architecturemodeling.launcher.validation.RAV " +
                "src/it/uni/sim/architecturemodeling/launcher/models/weaving/rawebbrowser-mozilla.model " +
                "src/it/uni/sim/architecturemodeling/launcher/metamodels/RA_ADL.ecore " +
                "src/it/uni/sim/architecturemodeling/launcher/models/referencearchitectures/smartparking.ecore"
            );
            System.exit(1);
        }

        modelPath = args[0];
        metamodelPath = args[1];
        referenceArchitecturePath = args[2];

        // 1) Register architecturemodeling.ecore
        registerMetamodelGlobally(
            "src/it/uni/sim/architecturemodeling/launcher/metamodels/architecturemodeling.ecore"
        );

        // 2) Register RA_ADL.ecore
        registerMetamodelGlobally(metamodelPath);

        // 3) Register reference architecture ecore
        registerMetamodelGlobally(referenceArchitecturePath);

        new RAV().execute();
    }

    @Override
    public IEolModule createModule() {
        return new EvlModule();
    }

    @Override
    public List<IModel> getModels() throws Exception {
        List<IModel> models = new ArrayList<IModel>();

        System.out.println("### [RAV.getModels] Loading archmm (architecturemodeling.ecore) ###");
        models.add(
            createEmfModelByURI(
                "archmm",
                "src/it/uni/sim/architecturemodeling/launcher/metamodels/architecturemodeling.ecore",
                "http://www.eclipse.org/emf/2002/Ecore",
                true,
                false
            )
        );

        System.out.println("### [RAV.getModels] Loading ramm (" + referenceArchitecturePath + ") ###");
        models.add(
            createEmfModelByURI(
                "ramm",
                referenceArchitecturePath,
                "http://www.eclipse.org/emf/2002/Ecore",
                true,
                false
            )
        );

        System.out.println("### [RAV.getModels] Loading weaving model via nsURI raadl ###");
        models.add(
            createEmfModelByURI(
                "weaving",
                modelPath,
                "http://it.uni.sim/ra_adl",
                true,
                true
            )
        );

        return models;
    }

    @Override
    public String getSource() throws Exception {
        return "src/it/uni/sim/architecturemodeling/launcher/validation/RA-ADL-validator.evl";
    }

    @Override
    public void postProcess() {
        EvlModule module = (EvlModule) this.module;

        Collection<UnsatisfiedConstraint> unsatisfied =
            module.getContext().getUnsatisfiedConstraints();

        if (unsatisfied.size() > 0) {
            System.err.println(unsatisfied.size() + " constraint(s) have not been satisfied");
            for (UnsatisfiedConstraint uc : unsatisfied) {
                System.err.println(uc.getMessage());
                for (FixInstance fix : uc.getFixes()) {
                    try {
                        fix.perform();
                    } catch (EolRuntimeException e) {
                        e.printStackTrace();
                    }
                }
            }
        } else {
            System.out.println("All constraints have been satisfied");
        }
    }


    private static void registerMetamodelGlobally(String ecorePath) {
        try {
            System.out.println("### [registerMetamodel] Registering: " + ecorePath);

            Registry globalReg = Resource.Factory.Registry.INSTANCE;
            Map<String, Object> map = globalReg.getExtensionToFactoryMap();
            map.put("ecore", new XMIResourceFactoryImpl());

            ResourceSet rs = new ResourceSetImpl();

            File f = new File(ecorePath);
            URI uri = URI.createFileURI(f.getAbsolutePath());

            Resource res = rs.getResource(uri, true);

            for (EObject o : res.getContents()) {
                if (o instanceof EPackage) {
                    EPackage p = (EPackage) o;
                    String nsURI = p.getNsURI();
                    System.out.println("[registerMetamodel] EPackage: " + p.getName() + " nsURI=" + nsURI);
                    if (nsURI != null && !nsURI.isEmpty()) {
                        EPackage.Registry.INSTANCE.put(nsURI, p);
                        System.out.println("[registerMetamodel] Put in global registry: " + nsURI);
                    }
                }
            }
        } catch (Exception ex) {
            System.err.println(">>> [registerMetamodel] Could not register " + ecorePath + " globally");
            ex.printStackTrace();
        }
    }
}