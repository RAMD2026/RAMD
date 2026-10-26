package it.uni.sim.architecturemodeling.launcher.validation;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

public class RawEMFTest {

    private static URI fileURI(String relativePath) {
        File f = new File(relativePath);
        System.out.println("[PATH] " + relativePath + " -> " + f.getAbsolutePath());
        return URI.createFileURI(f.getAbsolutePath());
    }

    public static void main(String[] args) {
        try {
            ResourceSet rs = new ResourceSetImpl();

            Map<String, Object> map = Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap();
            map.put("ecore", new XMIResourceFactoryImpl());
            map.put("model", new XMIResourceFactoryImpl());
            map.put("xmi",   new XMIResourceFactoryImpl());

            System.out.println("=== STEP 1: Load architecturemodeling.ecore ===");
            Resource archMMRes = rs.getResource(
                fileURI("src/it/uni/sim/architecturemodeling/launcher/metamodels/architecturemodeling.ecore"),
                true
            );
            for (EObject o : archMMRes.getContents()) {
                if (o instanceof EPackage) {
                    EPackage p = (EPackage) o;
                    System.out.println("  [archmm] EPackage: " + p.getName() + " nsURI=" + p.getNsURI());
                    rs.getPackageRegistry().put(p.getNsURI(), p);
                }
            }

            System.out.println("\n=== STEP 2: Load Ramm.ecore ===");
            Resource rammRes = rs.getResource(
                fileURI("src/it/uni/sim/architecturemodeling/launcher/models/referencearchitectures/Ramm.ecore"),
                true
            );
            for (EObject o : rammRes.getContents()) {
                if (o instanceof EPackage) {
                    EPackage p = (EPackage) o;
                    System.out.println("  [ramm] EPackage: " + p.getName() + " nsURI=" + p.getNsURI());
                    if (p.getNsURI() != null && !p.getNsURI().isEmpty()) {
                        rs.getPackageRegistry().put(p.getNsURI(), p);
                    }
                }
            }

            System.out.println("\n=== STEP 3: Load RA_ADL.ecore ===");
            Resource raadlRes = rs.getResource(
                fileURI("src/it/uni/sim/architecturemodeling/launcher/metamodels/RA_ADL.ecore"),
                true
            );
            for (EObject o : raadlRes.getContents()) {
                if (o instanceof EPackage) {
                    EPackage p = (EPackage) o;
                    System.out.println("  [raadl] EPackage: " + p.getName() + " nsURI=" + p.getNsURI());
                    rs.getPackageRegistry().put(p.getNsURI(), p);
                }
            }

            System.out.println("\n=== STEP 4: Load mozilla.model (architecture) ===");
            Resource archRes = rs.getResource(
                fileURI("src/it/uni/sim/architecturemodeling/launcher/models/architectures/mozilla.model"),
                true
            );
            printRootInfo("architecture(mozilla)", archRes);

            System.out.println("\n=== STEP 5: Load rawebbrowser-mozilla.model (weaving) ===");
            Resource weavingRes = rs.getResource(
                fileURI("src/it/uni/sim/architecturemodeling/launcher/models/weaving/rawebbrowser-mozilla.model"),
                true
            );
            printRootInfo("weaving(rawebbrowser-mozilla)", weavingRes);

            System.out.println("\n=== DONE: All models loaded without EMF exception ===");

        } catch (Exception ex) {
            System.err.println(">>> EXCEPTION while loading models:");
            ex.printStackTrace();
        }
    }

    private static void printRootInfo(String label, Resource res) {
        EList<EObject> contents = res.getContents();
        System.out.println("  [" + label + "] #root objects = " + contents.size());
        for (int i = 0; i < contents.size(); i++) {
            EObject root = contents.get(i);
            if (root != null && root.eClass() != null && root.eClass().getEPackage() != null) {
                System.out.println("    root[" + i + "]: " +
                    root.eClass().getName() +
                    " (nsURI=" + root.eClass().getEPackage().getNsURI() + ")");
            } else {
                System.out.println("    root[" + i + "]: " + root);
            }
        }
    }
}
