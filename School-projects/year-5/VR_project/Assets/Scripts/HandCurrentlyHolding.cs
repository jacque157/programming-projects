using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class HandCurrentlyHolding : MonoBehaviour
{
    [SerializeField]
    private IXRSelectInteractable currentInteractable;

    public IXRSelectInteractable CurrentInteractable
    {
        get { return currentInteractable; }
    }

    public void OnSelectEnter(SelectEnterEventArgs args)
    {
        currentInteractable = args.interactableObject;
        Debug.Log("Object attached: " + currentInteractable.transform.gameObject.name);
    }

    public void OnSelectExit(SelectExitEventArgs args)
    {
        //Debug.Log("Object detached: " + currentInteractable.transform.gameObject.name);
        currentInteractable = null;
        var r =  args.interactableObject.transform.gameObject.GetComponent<Rigidbody>();
        if (r)
        {
            r.isKinematic = false;
            r.detectCollisions = true;
        }
    }
    public void OnSelectExitInventory(SelectExitEventArgs args)
    {
        OnSelectExit(args);
        //get currently grabbed object
        var r =  args.interactableObject.transform.gameObject.GetComponent<Rigidbody>();
        if (r)
        {
            r.isKinematic = true;
            r.detectCollisions = false;
        }
    }
}