using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class ChestContent : MonoBehaviour
{
    public List<XRGrabInteractable> storedItems = new List<XRGrabInteractable>();
    // Start is called before the first frame update
    void Start()
    {
        foreach (var i in storedItems)
        {
            i.enabled = false;
        }
    }
    //enable grabbable items in chest to be grabbed(to prevent grabbing through walls) 
    void ShowReward()
    {
        foreach (var i in storedItems)
        {
            i.enabled = true;
        }
    }
    


}
