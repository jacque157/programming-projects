using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;

public class PickUp : MonoBehaviour
{
     public InputActionReference selected;
     public GameObject selectedObbject;
     public XRDirectInteractor rightHandDirectInteractor;

     [SerializeField]
     public List<IXRSelectInteractable> listSelected = new List<IXRSelectInteractable>();

     public int selectedItemsCount = 0;
     public bool isSelecting = false;

    // Start is called before the first frame update
    void Start()
    {
        selected.action.performed += OnSelect;
        selected.action.canceled += OnDeSelect;
        selectedItemsCount = 0;
    }

    // Update is called once per frame
    void Update()
    {
        //spravit event na inventar

        selectedItemsCount = rightHandDirectInteractor.interactablesSelected.Count;
        listSelected = rightHandDirectInteractor.interactablesSelected;
        isSelecting = rightHandDirectInteractor.hasSelection;
    }

    void OnSelect(InputAction.CallbackContext context)
    {
        Debug.Log("onselect"+rightHandDirectInteractor.hasSelection);
        //if not already grabbing something
        if (selectedObbject == null &&  rightHandDirectInteractor.interactablesSelected.Count >0)
        {
            //get currently grabbed object
            selectedObbject = rightHandDirectInteractor.interactablesSelected[0].transform.gameObject;
            var r = selectedObbject.GetComponent<Rigidbody>();
            if (r)
                r.detectCollisions = false;
        }
    }
    
    void OnDeSelect(InputAction.CallbackContext context)
    {
        if (!rightHandDirectInteractor.hasSelection)
            selectedObbject = null;
    }
}
