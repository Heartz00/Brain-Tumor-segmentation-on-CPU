def process_uploaded_zip(uploaded_zip):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save and extract zip
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            # Initialize files dictionary
            files = {
                't1n': None, 't1c': None, 
                't2f': None, 't2w': None,
                'seg': None
            }
            
            # First, check if we have the nested folder structure
            base_path = tmpdir
            nested_folder = os.path.join(tmpdir, "data_for test")
            if os.path.exists(nested_folder):
                base_path = nested_folder
                # Check for double nesting (your case shows "data_for test/data_for test/")
                double_nested = os.path.join(nested_folder, "data_for test")
                if os.path.exists(double_nested):
                    base_path = double_nested
            
            # Define matching patterns
            patterns = {
                't1n': ['-t1n.', '-t1n_', 't1n.nii', 't1_native'],
                't1c': ['-t1c.', '-t1c_', 't1c.nii', 't1_contrast'],
                't2f': ['-t2f.', '-t2f_', 't2f.nii', 'flair'],
                't2w': ['-t2w.', '-t2w_', 't2w.nii', 't2_weighted'],
                'seg': ['-seg.', '_seg.', 'seg.nii', 'label']
            }
            
            # Search through all files in the base directory
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    f_lower = f.lower()
                    if f.endswith('.nii.gz') or f.endswith('.nii'):
                        # Check for your specific pattern
                        if '-t1n' in f_lower: files['t1n'] = os.path.join(root, f)
                        elif '-t1c' in f_lower: files['t1c'] = os.path.join(root, f)
                        elif '-t2f' in f_lower: files['t2f'] = os.path.join(root, f)
                        elif '-t2w' in f_lower: files['t2w'] = os.path.join(root, f)
                        elif '-seg' in f_lower: files['seg'] = os.path.join(root, f)
            
            # Verify we found all required files (seg is optional)
            required_files = ['t1n', 't1c', 't2f', 't2w']
            missing = [ft for ft in required_files if files[ft] is None]
            
            if missing:
                st.error(f"Missing required scan files: {', '.join(missing)}")
                st.info("Files found in ZIP:")
                for root, _, filenames in os.walk(base_path):
                    for f in filenames:
                        if f.endswith('.nii.gz') or f.endswith('.nii'):
                            st.info(f"- {os.path.join(root, f)}")
                return None
            
            return files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return None
