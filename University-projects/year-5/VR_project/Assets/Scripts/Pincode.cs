using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Interaction.Toolkit;

//for now use only 4-number codes
public class Pincode : MonoBehaviour
{
    public List<DialChange> pincodeHolders = new List<DialChange>();

    public string correctPass = "1234";

    public UnityEvent CorrectPinAction;
    public bool isCorrect = false;

#if UNITY_EDITOR
    public bool debugEditor = false;
# endif
     
    // Start is called before the first frame update
    void Start()
    {
        CheckStringOnStart();
    }

    // Update is called once per frame
    void Update()
    {
        #if UNITY_EDITOR
        if (Input.GetKeyDown(KeyCode.O)) 
        {
            CheckCode(null);
        }
        #endif
    }

    public void CheckCode(SelectExitEventArgs args)
    {
        //NOTE ADD this as event in the dialchange...
        string r="";
        for (int i = 0; i < pincodeHolders.Count; i++)
        {
            var val = Mathf.RoundToInt(pincodeHolders[i].GetAngle()/36.0f);
            val %= 10;
            r += val.ToString();
        }
        #if UNITY_EDITOR
        if(debugEditor)
            Debug.Log("obtained code " + r);
        # endif
        if (r.Equals(correctPass))
        {
            CorrectPinAction.Invoke();
            isCorrect = true;
        }
        isCorrect = false;

    }

    void CheckStringOnStart()
    {
        bool errr = false;
        if (correctPass.Length < 4)
        {
            correctPass = "0000";
            errr = true;
        }
        if (correctPass.Length > 4)
        {
            correctPass = "0000";
            errr = true;
        }
        if(errr)
            Debug.LogError("correctPass was of incorrect size, reset to 0000");
    }
}
