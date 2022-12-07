
linear(a::Real,b::Real,t::Real) = a * t +b
linear(A::AbstractVector, B::AbstractVector, t::Real) = A * t + B

function train_test_split(D::ExternalInputsDataset, TP_loc::Int)
    train_data = D.X[1:TP_loc,:]
    test_data = D.X[TP_loc:end,:]
    train_time = D.S[1:TP_loc,:]
    test_time = D.S[TP_loc:end,:]
    D_train = ExternalInputsDataset(train_data, train_time, "train")
    D_test = ExternalInputsDataset(test_data, test_time, "test")
    return D_train, D_test
end

function get_model_from_path(path::String)
    if occursin("ShrinkingLorenz", path)
        model = "ShrinkingLorenz"
    elseif occursin("StopBurstBN", path)
        model = "StopBurstBN"
    elseif occursin("RampUpBN", path)
        model = "RampUpBN"
    elseif occursin("PaperLorenzBigChange", path)
        model = "PaperLorenzBigChange"
    elseif occursin("PaperLorenzSmallChange", path)
        model = "PaperLorenzSmallChange"
    else
        throw("Not implemented")
    end
    return model
end