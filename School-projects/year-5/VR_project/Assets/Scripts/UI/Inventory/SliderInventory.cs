using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Serialization;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Inputs.Simulation;

//Currently its looking at the hand and doesnt rotate if head rotates
public class SliderInventory : MonoBehaviour
{
    public List<Image> panels = new();
    public List<GameObject> slots = new();
    public Image overlayPanel;

    private Vector3 startHandPos;
    private Vector3 startHandPosOffset;
    private Vector3 curHandPos;
    private Vector3 hudRotation;
    private Quaternion startUIRot;
    private Quaternion offsetRotation;
    private RectTransform UIPanel;


    private Camera cam;
    private UnityEngine.XR.InputDevice device;
    public InputActionProperty UIPressed;
    public InputActionProperty RightHandPosition;
    public InputActionProperty PlayerPosition;
    public InputActionProperty RightHandRotation;
    public InputActionProperty PlayerRotation;

    private bool inInventory = false;
    public float minDiff = 0.1f;
    public float minHandOffsetToAllowPlaneRotation = 0.02f;
    public float maxHandOffsetToAllowPlaneRotation = 0.15f;
    public int selection = -1;

    //internal check to correct for position of hands and player when using device sim, it still breaks if xr origin os not at (0,0,0) though...
    public bool isSimulator = false;
    public float startHandPosForwardOffset = 0.1f;
    public XROrigin xrOrigin;
    public ActionBasedController rightHand;
    [SerializeField] private float difX;
    [SerializeField] private float difY;

    public bool rotateWithHead = false;

    public bool enableFreeGrab = false; //require pressing grab to get item from inventory

    private InteractionLayerMask ignoreGrabLayers;
    
    public List<bool> rotationLimits = new(3);

    public HandCurrentlyHolding handCurrentlyHolding;
    public XRInteractionManager xrInteractionManager;

    private void Start()
    {
        UIPanel = GetComponent<RectTransform>();
        inInventory = false;
        UIPanel.gameObject.SetActive(inInventory);
        cam = Camera.main;
        //slots = new List<GameObject>(panels.Count); //do it in inspector

        //find head and hand if not assigned
        if (!xrOrigin)
            xrOrigin = FindAnyObjectByType<XROrigin>();
        xrInteractionManager = FindObjectOfType<XRInteractionManager>();
        if (!rightHand)
        {
            var hands = FindObjectsOfType<XRInteractionGroup>();
            foreach (var h in hands)
                if (h.groupName.Equals("Right"))
                {
                    rightHand = h.GetComponent<ActionBasedController>();
                    if (!handCurrentlyHolding)
                    {
                        handCurrentlyHolding = h.GetComponentInChildren<HandCurrentlyHolding>();
                        if (!handCurrentlyHolding)
                            Debug.LogError("couldnt find HandCurrentlyHolding on the right hand!");
                    }

                    break;
                }
        }


        var o = FindAnyObjectByType<XRDeviceSimulator>();
        if (o != null)
            if (o.gameObject.activeSelf)
                isSimulator = true;

        overlayPanel.gameObject.SetActive(false);


        UIPressed.action.performed += OnInventoryEnter;
        UIPressed.action.canceled += OnInventoryCancel;
    }

    private void Update()
    {
        if (inInventory)
        {
            //Vector3 pointA= PlayerPosition
            curHandPos = GetHandPos();
            var curPlayerPos = GetPlayerPos();
            var curPlayerRot = GetPlayerRot();

            //Debug.Log("playerpos "+curPlayerPos);

            //dalsi problem, ak origin nezacne byt na 0,0 tak to aj tak nesedi s poziciou....
            var rotationDifference = new Quaternion();
            var rotatedOffset = new Vector3();


            //todo fix tilt of head?
            //use this to ignore up/down rotation of the player 
            //rotationDifference = Quaternion.Inverse(offsetRotation) * Quaternion.Euler(1,curPlayerRot.eulerAngles.y ,curPlayerRot.eulerAngles.z);
            rotatedOffset = rotationDifference * startHandPosOffset;

            UIPanel.transform.position = curPlayerPos + rotatedOffset;

            //only if far enough from the ui
            //to cancell fast jittery rotations close to the object
            var newFront = GetHandRot() * Vector3.back;
            newFront = curHandPos + newFront * 0.2f;

            // Check the sign of the dot product
            //var image = UIPanel.GetComponent<Image>();
            var newRight = curPlayerRot * Vector3.right;
            var newUp = curPlayerRot * Vector3.up;
            difX = Vector3.Dot(UIPanel.transform.position - curHandPos, newRight);
            difY = Vector3.Dot(UIPanel.transform.position - curHandPos, newUp);

            //always reset
            selection = -1;

            if (math.abs(difX) > math.abs(difY) && math.abs(difX) > minDiff)
            {
                if (difX < 0)
                    //left
                    //image.color = Color.blue;
                    selection = 0;
                else
                    //right
                    //image.color = Color.red;
                    selection = 1;
            }
            else if (math.abs(difY) > minDiff)
            {
                if (difY > 0)
                    //bottom
                    //image.color = Color.green;
                    selection = 2;
                else
                    //top
                    //image.color = Color.yellow;
                    selection = 3;
            }

            //overlayPanel.rectTransform.position = panels[selection].rectTransform.position;
            //nah better, parent it and reset rot and trans
            if (selection != -1)
            {
                overlayPanel.gameObject.SetActive(true);
                overlayPanel.rectTransform.SetParent(panels[selection].rectTransform);
                overlayPanel.rectTransform.position = panels[selection].rectTransform.position;
                overlayPanel.rectTransform.rotation = panels[selection].rectTransform.rotation;

                /*if(handCurrentlyHolding.CurrentInteractable !=null)
                {
                    handCurrentlyHolding.CurrentInteractable.transform.parent = panels[selection].rectTransform;
                    handCurrentlyHolding.CurrentInteractable.transform.position = Vector3.zero;
                }*/
            }
            else
            {
                overlayPanel.gameObject.SetActive(false);
            }

            if (Vector3.Distance(curHandPos, startHandPos) > minHandOffsetToAllowPlaneRotation
                && math.abs(difX) < maxHandOffsetToAllowPlaneRotation &&
                math.abs(difY) < maxHandOffsetToAllowPlaneRotation)
                UIPanel.LookAt(curHandPos);
        }
    }

    //select or add object
    private void SelectObject()
    {
        Debug.Log("selection " + selection);

        if (handCurrentlyHolding.CurrentInteractable != null)
        {
            //we are adding item to the inventory from hand
            Debug.Log("we are adding item to the inventory from hand");

            GameObject objToDrop = null;
            //check if slot is empty
            if (slots[selection] != null)
            {
                if(!enableFreeGrab)
                    return;
                Debug.Log("we need to swap");
                //swap item already at hand
                objToDrop = slots[selection];
            }

            //add object to selection
            handCurrentlyHolding.CurrentInteractable.transform.parent = panels[selection].rectTransform;
            handCurrentlyHolding.CurrentInteractable.transform.localPosition = Vector3.zero;
            handCurrentlyHolding.CurrentInteractable.transform.localRotation = Quaternion.identity;

            //TODO set scale to match the UI square
            //but we need to remember that slots size to revert it

            slots[selection] = handCurrentlyHolding.CurrentInteractable.transform.gameObject;
            //dettach
            //var d = rightHand.GetComponent<XRDirectInteractor>();
            //var f = handCurrentlyHolding.CurrentInteractable.AllowSe;
            var args = new SelectExitEventArgs();
            args.interactableObject = handCurrentlyHolding.CurrentInteractable;
            //args.interactorObject = handCurrentlyHolding
            handCurrentlyHolding.OnSelectExitInventory(args);
            //disable grabbing
            //handCurrentlyHolding.CurrentInteractable.transform.gameObject.GetComponent<XRGrabInteractable>()
            //    .enabled = false;
            //lastly drop the previously selected obj
            if(objToDrop)
                DropObj(objToDrop);

        }
        else
        {
            if (slots[selection] == null)
                return;
            if(!enableFreeGrab)
                return;
            Debug.Log("hand is free adding item from inventory");
            DropObj(slots[selection]);
            slots[selection] = null;

        }
    }

    public void DropObj(GameObject o)
    {
        //we are taking item from the inventory, nothing in hand
        //check if its not empty
        //drop currently selected item to swap with the new one
        //o = handCurrentlyHolding.CurrentInteractable.transform.gameObject;
        //dettach
        //var d = rightHand.GetComponent<XRDirectInteractor>();
        //var f = handCurrentlyHolding.CurrentInteractable.AllowSe;

        //o.transform.parent = rightHand.transform;
        //o.transform.localPosition = Vector3.zero;
        o.transform.SetParent(null);
        Debug.Log("dropping "+o.name);
        var args = new SelectEnterEventArgs();
        //TODO je tu bug ked vyberiem vec z inventaru, nemusim drzat grab
        //TODO potom vyberiem dalsiu vec z inventaru a obe su v ruke naraz!
        args.interactableObject = o.GetComponent<XRGrabInteractable>();
        args.interactorObject = rightHand.gameObject.GetComponent<XRDirectInteractor>();
        args.manager = xrInteractionManager;
        //force interaction
        xrInteractionManager.SelectEnter(args.interactorObject, args.interactableObject);

        //handCurrentlyHolding.OnSelectEnter(args);
        //args.interactorObject.OnSelectEntered(args);

        //we still want it to be off?
        var r =  o.GetComponent<Rigidbody>();
        if (r != null)
        {
            r.isKinematic = false;
            r.detectCollisions = true;
        }
    }

    private void OnInventoryEnter(InputAction.CallbackContext context)
    {
        //we want to get an item
        //workaround, make sure all rigidbodies are sleeping

        for (var i = 0; i < slots.Count; i++)
            if (slots[i] != null)
            {
                var r = slots[i].GetComponent<Rigidbody>();
                if (r!=null)
                {
                    r.isKinematic = true;
                    r.detectCollisions = false;
                }
            }

        startHandPos = GetHandPos();
        //add slight offset make it to the back
        startHandPos += GetHandRot() * Vector3.forward * startHandPosForwardOffset;
        startHandPosOffset = startHandPos - GetPlayerPos();
        //UIPanel.LookAt(curHandPos); //TODO disable for now for testing
        //now store the rot
        var rotation = Quaternion.LookRotation(startHandPos - UIPanel.transform.position);
        startUIRot = rotation;

        //maybe remember start rotation and then make limits
        //Debug.Log("start pos" + startHandPos);
        //Debug.Log("start posOffset" + startHandPosOffset);

        offsetRotation = GetPlayerRot();
        //Debug.Log("on");
        inInventory = true;
        UIPanel.gameObject.SetActive(inInventory);
    }

    private void OnInventoryCancel(InputAction.CallbackContext context)
    {
        //only if it was performed, ignore quick cancels
        ExitInventory();
    }

    public void ExitInventory()
    {
        if (selection != -1 && inInventory)
            SelectObject();
        //if active, stop ui
        inInventory = false;
        UIPanel.gameObject.SetActive(inInventory);
        overlayPanel.gameObject.SetActive(false);
    }

    private Vector3 GetPlayerPos()
    {
        return xrOrigin.Camera.transform.position;
        // return PlayerPosition.action.ReadValue<Vector3>();
    }

    private Quaternion GetPlayerRot()
    {
        return xrOrigin.Camera.transform.rotation;
    }

    private Vector3 GetHandPos()
    {
        return rightHand.transform.position;
        //return RightHandPosition.action.ReadValue<Vector3>();
    }

    private Quaternion GetHandRot()
    {
        return rightHand.transform.rotation;
    }
}