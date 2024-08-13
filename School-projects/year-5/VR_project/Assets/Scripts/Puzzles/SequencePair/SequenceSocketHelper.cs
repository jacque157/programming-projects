using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Content.Interaction;
using UnityEngine.XR.Interaction.Toolkit;

public class SequenceSocketHelper : MonoBehaviour
{
    public SequenceIdData sequenceIdData;

    public XRSocketInteractor xrSocketInteractor;
    // Start is called before the first frame update
    void Start()
    {
        xrSocketInteractor = GetComponent<XRSocketInteractor>();
    }



    public bool CheckPair()
    {
        if (!xrSocketInteractor)
            return false;
        if (xrSocketInteractor.hasSelection)
        {
            if (xrSocketInteractor.interactablesSelected[0].transform.GetComponent<SequenceItem>().sequenceIdData == sequenceIdData)
            {
                //Debug.Log("correct pairing found");
                return true;
            }
        }
        
        return false;
    }
}
