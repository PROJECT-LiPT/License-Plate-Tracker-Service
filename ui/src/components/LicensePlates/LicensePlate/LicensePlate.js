import React, { useEffect } from 'react';
import Draggable from 'react-draggable';

import './LicensePlate.css';

const LicensePlate = ({currentBase64}) => {
  const onCloseForm = () => {
    //
  }
  return (
    <Draggable bounds="parent">
      <div id="menu" className="lp_container_menu">
        <div className="title_bar">
           LP ID 
        </div>
        <div className="user_info">
          <div> Owner: { currentBase64 ? <span> Mr. A </span> : <span> Loading...</span> } </div>
          <div> Age: { currentBase64 ? <span> 28 </span> : <span> Loading...</span> } </div>
          <div> Gender: { currentBase64 ? <span> Male </span> : <span> Loading...</span> }</div>
          <div> Register location: { currentBase64 ? <span> Mars </span> : <span> Loading...</span> } </div>
        </div>
        <div className="lp_image_container">
          { currentBase64 ? <img className="lp_image" src={`${currentBase64}`}/> : <span> Loading...</span> }
        </div>
      </div>
    </Draggable>
  );
};
export default LicensePlate;