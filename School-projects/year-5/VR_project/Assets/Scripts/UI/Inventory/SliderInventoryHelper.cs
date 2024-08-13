using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Inputs.Simulation;

//Old script to test without XROrigin 
//Currently its looking at the hand and doesnt rotate if head rotates
public class SliderInventoryHelper : MonoBehaviour
{
    [SerializeField] private GameObject handToTrack;
    [SerializeField] private GameObject HMDToTrack;
    [SerializeField] private GameObject UIPanel;
    private Vector3 startHandPos;
    private Vector3 startHandPosOffset;
    private Vector3 curHandPos;
    private Vector3 hudRotation;
    public float minDiff = 0.1f;
    public float minHandOffsetToAllowRotation = 0.01f;
    private Quaternion offsetRotation;
    public float difX;
    public float difY;
    int selection = -1;
    // Start is called before the first frame update
    private void Start()
    {
        startHandPos = handToTrack.transform.position;
        startHandPosOffset = startHandPos - HMDToTrack.transform.position;
        
        offsetRotation = HMDToTrack.transform.rotation;
        //Debug.Log("on");
        UIPanel.gameObject.SetActive(true);

    }

    // Update is called once per frame
    private void Update()
    {
        //Vector3 pointA= PlayerPosition
            curHandPos =  handToTrack.transform.position;
            var curPlayerPos = HMDToTrack.transform.position;

            Quaternion rotationDifference = new Quaternion();
            Vector3 rotatedOffset = new Vector3();
            //only if far enough from the ui

            rotationDifference = Quaternion.Inverse(offsetRotation) * HMDToTrack.transform.rotation;
            rotatedOffset = rotationDifference * startHandPosOffset;

            UIPanel.transform.position = curPlayerPos + rotatedOffset;

            //to cancell fast jittery rotations close to the object
            if (Vector3.Distance(curHandPos, startHandPos) > minHandOffsetToAllowRotation)
                UIPanel.transform.LookAt(curHandPos);
            

            var image = UIPanel.GetComponent<Image>();
            // Check the sign of the dot product

            Vector3 newRight =  HMDToTrack.transform.rotation * Vector3.right; //Vector3.Cross(curPlayerPos)
            Vector3 newUp =  HMDToTrack.transform.rotation * Vector3.up; //Vector3.Cross(curPlayerPos)
            difX = Vector3.Dot(UIPanel.transform.position - curHandPos, newRight);
            difY = Vector3.Dot(UIPanel.transform.position - curHandPos, newUp);

            
            //difX = Vector3.Dot(UIPanel.transform.position - curHandPos, HMDToTrack.transform.right);
            //difY = Vector3.Dot(UIPanel.transform.position - curHandPos, HMDToTrack.transform.up);


            //if( curHandPos. - UIPanel.transform.position.x)

            image.color = Color.white;
            selection = -1;
            //TODO xko takto kontrolovat nie je dobre lebo mozme mat rozne rotacie

            if (math.abs(difX) > math.abs(difY) && math.abs(difX) > minDiff)
            {
                if (difX <0)
                {
                    image.color = Color.blue;
                    selection = 0;
                }
                else
                {
                    image.color = Color.red;
                    selection = 1;
                }
            }
            else if (math.abs(difY) > minDiff)
            {
                if (difY < 0)
                {
                    image.color = Color.green;
                    selection = 2;
                }
                else
                {
                    image.color = Color.yellow;
                    selection = 3;
                }
            }

    }
}
