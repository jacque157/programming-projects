using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

//This was a test on how to handle activate event on objects with behaviour
// either we add the listeners with script
//or manually in the onspector(more cumbersome imho)
public class TestActivate : MonoBehaviour
{
    public XRBaseInteractable interactable;
    // Start is called before the first frame update
    void Start()
    {
        interactable = GetComponent<XRBaseInteractable>();
        interactable.selectEntered.RemoveAllListeners();
        interactable.activated.RemoveAllListeners();
        Debug.LogWarning("all activate and select listeners removed at start on this obj!");
        //interactable.selectEntered.AddListener();
        interactable.activated.AddListener(OnActivate);

    }

    
    
    // Update is called once per frame
    void Update()
    {
        
    }

    //Activate action only works, when already selected
    public void OnActivate(ActivateEventArgs a)
    {
        Debug.Log("I was activated");
        //can we check if holding
        //a.interactableObject.transform.name
        Debug.Log("Is sel " + interactable.isSelected);

    }
    public void OnSelect()
    {
        Debug.Log("I was selected");
        
        
    }
    public void OnSelectEnter(SelectEnterEventArgs args)
    {
        //currentInteractable = args.interactableObject;
        Debug.Log("Object attached: " + interactable.gameObject.name);
    }
}
